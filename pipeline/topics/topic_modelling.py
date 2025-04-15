from collections import Counter

import click
import numpy as np
import pyarrow as pa
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from pipeline.topics.utils import EmptyClusterModel
from pipeline.topics.utils import EmptyDimensionalityReduction
from pipeline.utils.knn import KNN


class TopicModel(KNN):
    """Methods to perform topic modelling. Can be used with existing UMAP embeddings
    or cluster labels.
    """

    def __init__(
        self,
        input_bucket,
        embedding_col,
        umap_embedding_col,
        cluster_col,
        additional_cols=None,
        cluster_ids=None,
        topic_model=None,
    ):
        """Initialise parent class and parameters.

        Args:
            input_bucket(list, str): S3 URI or list of S3 URIs for text embedding directory.
            embedding_col(str): Column containing embeddings.
            umap_embedding_col(str): Column containing UMAP embeddings.
            cluster_col(str): Column containing cluster labels.
            cluster_ids(list): If provided, only performs topic modelling on the given clusters.
            topic_model(str, BERTopic): Existing topic model (optional).

        """
        if not isinstance(additional_cols, list):
            additional_cols = []
        additional_cols.extend(["title", "abstract", umap_embedding_col, cluster_col])

        schema = pa.schema(
            [
                ("publication_id", pa.string()),
                ("abstract", pa.string()),
                ("title", pa.string()),
                (embedding_col, pa.large_list(pa.float64())),
                (umap_embedding_col, pa.large_list(pa.float64())),
                (cluster_col, pa.int32()),
            ]
        )

        super().__init__(
            input_bucket, embedding_col, additional_cols=additional_cols, schema=schema
        )

        self.data[cluster_col] = self.data[cluster_col].fillna(-1)

        if cluster_ids is not None:
            self.data = self.data[self.data[cluster_col].isin(cluster_ids)]
        self._umap_embedding_col = umap_embedding_col
        self._cluster_col = cluster_col

        if isinstance(topic_model, str):
            self.topic_model = BERTopic.load(topic_model)
        else:
            self.topic_model = topic_model

    def update_cluster_ids(self, cluster_id_map):
        """Update cluster ID column with new cluster IDs.

        Args:
            cluster_id_map(dict): New cluster IDs.
        """
        self.data[self._cluster_col] = self.data.index.map(cluster_id_map)

    def run_bertopic(
        self,
        dimensionality_model=None,
        cluster_model=None,
        vectorizer_model=None,
        ctfidf_model=None,
    ):
        """Run topic modelling pipeline.

        Args:
            dimensionality_model(Any): Dimensionality reduction model (e.g. UMAP). Can be any model as long as it has
                fit and transform methods. If 'None', uses empty model by default.
            cluster_model(Any): Cluster model (e.g. HDBSCAN, KMeans). Can be any model as long as it has
                fit and predict methods. If 'None', uses empty model by default.
            vectorizer_model(Any): Vectorizer model (e.g. count vectorizer). If 'None', uses sklearn CountVectorizer
                with English stopwords by default.
            ctfidf_model(Any): cTFIDF model. If 'None', uses BERTopic ClassTfidfTransformer by default.
        """
        if dimensionality_model is None:
            dimensionality_model = EmptyDimensionalityReduction(
                np.array(self.data[self._umap_embedding_col].tolist())
            )
        if cluster_model is None:
            cluster_model = EmptyClusterModel(self.data[self._cluster_col].tolist())
        if vectorizer_model is None:
            vectorizer_model = CountVectorizer(stop_words="english", min_df=0.1)
        if ctfidf_model is None:
            ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.topic_model = BERTopic(
            umap_model=dimensionality_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            verbose=True,
        )
        topics, _ = self.topic_model.fit_transform(
            self.data["abstract"].tolist(),
            np.array(self.data[self._embedding_col].tolist()),
        )
        self.data["topics"] = topics
        self.topic_model = self.topic_model

    def update_topics(self, new_topics):
        """Update topic representation with new topic IDs by re-running vectorizer and cTFIDF models.

        Args:
            new_topics(list): New topic IDs to update topic model with.
        """
        print("\nUpdating topics...")
        self.topic_model.update_topics(
            self.data["abstract"].tolist(),
            topics=new_topics,
            vectorizer_model=self.topic_model.vectorizer_model,
            ctfidf_model=self.topic_model.ctfidf_model,
        )

    def merge_topics(self, topics_to_merge):
        """Reduce the number of topics by merging topics.

        Args:
            topics_to_merge(list): IDs of topics to merge.
        """
        self.topic_model.merge_topics(self.data["abstract"].tolist(), topics_to_merge)

    def _reduce_outliers_bertopic(self, threshold=0.85, **kwargs):
        """Reduce outliers with BERTopic.

        Args:
            threshold(float): Minimum distance or similarity when matching outlier documents with non-outlier topics.
            **kwargs: Additional arguments to be passed to BERTopic outlier reduction.
        """
        print("\nRunning outlier reduction with BERTopic...")
        new_topics = self.topic_model.reduce_outliers(
            self.data["abstract"].tolist(),
            self.data["topics"].tolist(),
            embeddings=np.array(self.data[self._embedding_col].tolist()),
            strategy="embeddings",
            threshold=threshold,
            **kwargs,
        )
        print(
            f"Coverage after outlier reduction: {sum(new_topics != -1)/len(new_topics) * 100}%"
        )
        self.data["topics_ored"] = new_topics

    def _reduce_outliers_knn(self, threshold, path_to_deduped_files=None):
        """
        Reduce outliers via k-nearest neighbours. For each outlier embedding, the most common non-outlier
        topic will be selected among its nearest neighbours with a similarity above the given threshold.

        Args:
            threshold(float): Minimum distance or similarity when matching outlier documents with non-outlier topics.
            path_to_deduped_files(str): s3 path to parquet files containing deduplicated indices.
        """
        print("\nRunning outlier reduction via k-nearest neighbours...")
        all_topics = np.array(self.data["topics"].tolist())
        embeddings = np.array(self.data[self._embedding_col].tolist())

        embeddings = self.dedupe_embeddings(
            embeddings=embeddings, source_fpath=path_to_deduped_files
        )
        all_topics = all_topics[self._dedupe_index]

        outlier_indices = [i for i, t in enumerate(all_topics) if t == -1]
        non_outlier_indices = [i for i, t in enumerate(all_topics) if t != -1]
        non_outlier_topics = np.array([t for t in all_topics if t != -1])

        knn_ind, knn_dist = self.faiss_knn(
            embeddings_train=embeddings[non_outlier_indices],
            embeddings_search=embeddings[outlier_indices],
        )

        outlier_topics = []
        for i in tqdm(range(len(outlier_indices))):
            neighbours = knn_ind[i][knn_dist[i] <= (1 - threshold)]
            if len(neighbours) > 0:
                neighbour_topics = Counter(non_outlier_topics[neighbours])
                outlier_topics.append(neighbour_topics.most_common(1)[0][0])
            else:
                outlier_topics.append(-1)
        for out_i, out_t in zip(outlier_indices, outlier_topics):
            all_topics[out_i] = out_t

        all_topics = all_topics[self._dedupe_inverse]

        print(
            f"Coverage after outlier reduction: {sum(all_topics != -1)/len(all_topics) * 100}%"
        )
        self.data["topics_ored"] = all_topics

    def reduce_outliers(
        self,
        strategy="knn",
        threshold=0.85,
        update_topics=False,
        path_to_deduped_files=None,
        **kwargs,
    ):
        """
        Run automatic outlier reduction. If k-nearest neighbour indices and distances are present in the
        dataset, this will be used for outlier reduction. Otherwise the BERTopic implementation will be used.

        Args:
            strategy(str): Run outlier reduction via k-nearest neighbours ("knn") or bertopic ("bertopic").
            threshold(float): Minimum distance or similarity when matching outlier documents with non-outlier topics.
            update_topics(bool): Whether to update the topic representation with newly assigned topic labels.
            path_to_deduped_files(str): s3 path to parquet files containing deduplicated indices.
            **kwargs: Additional arguments to be passed to BERTopic outlier reduction.
        """
        if strategy == "knn":
            self._reduce_outliers_knn(threshold, path_to_deduped_files)
        elif strategy == "bertopic":
            self._reduce_outliers_bertopic(threshold, **kwargs)

        if update_topics:
            self.update_topics(self.data["topics_ored"].tolist())

    def to_outlier(self, topics):
        """Manually set topics to outliers and update topic model.

        Args:
            topics(list): List of topic IDs to set to outliers.
        """
        reduced_topics = []
        for t in self.topic_model.topics_:
            if t in topics:
                reduced_topics.append(-1)
            else:
                reduced_topics.append(t)
        self.data[self._cluster_col] = reduced_topics
        self.run_bertopic()

    def save_topics(self, topic_col="topics"):
        """Save topic IDs by attaching them as a column to original parquet files.

        Args:
            topic_col(str): Column containing topic IDs.
        """
        print("\nAppending topics to parquet...")
        self.save(topic_col)

    def save_topic_model(self, fpath):
        """Save topic model (pickle).

        Args:
            fpath(str): Filename/path to save topic model to.
        """
        self.topic_model.save(fpath, serialization="pickle")


@click.command()
@click.option("--embedding_column", default="embeddings")
@click.option("--umap_embedding_column", default="umap_5dim_15n")
@click.option("--cluster_column", default="hdbscan_labels_25mcs_50ms")
@click.option("--fpath", default="data/topic_model")
@click.option(
    "--input_s3_uri",
    "-i",
    default="s3://datalabs-data/funding_impact_measures/embeddings_sample/embeddings.parquet",
)
@click.option("--reduce_outliers", is_flag=True)
def topic_modelling(
    embedding_column,
    umap_embedding_column,
    cluster_column,
    fpath,
    input_s3_uri,
    reduce_outliers,
):
    input_s3_paths = [fpath.strip() for fpath in input_s3_uri.split(",")]

    topic_model = TopicModel(
        input_bucket=input_s3_paths,
        embedding_col=embedding_column,
        umap_embedding_col=umap_embedding_column,
        cluster_col=cluster_column,
    )

    topic_model.run_bertopic()
    topic_model.save_topic_model(fpath)
    topic_model.save_topics()

    if reduce_outliers:
        topic_model.reduce_outliers(update_topics=True)
        topic_model.save_topics("topics_ored")
        topic_model.save_topic_model(f"{fpath}_ored")


if __name__ == "__main__":
    topic_modelling()
