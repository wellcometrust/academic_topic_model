import awswrangler as wr
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from pipeline.utils.base import BaseModel


class KNN(BaseModel):
    "Methods to calculate k-nearest neighbours using Faiss."

    def __init__(
        self,
        input_bucket,
        embedding_col="embeddings",
        additional_cols=None,
        schema=None,
    ):
        """Initialise parent class and parameters.

        Args:
            input_bucket(list, str): S3 URI or list of S3 URIs for text embedding directory.
            embedding_col(str): Column containing embeddings.
            additional_cols(Optional[list]): Names of any additional columns to load.
            schema(pa.schema): Pyarrow schema for embeddings data.

        """
        super().__init__(
            input_bucket=input_bucket,
            embedding_col=embedding_col,
            additional_cols=additional_cols,
            schema=schema,
        )
        self._dedupe_index = np.array(range(len(self.data)))
        self._dedupe_inverse = np.array(range(len(self.data)))

    def dedupe_embeddings(self, embeddings, source_fpath, dest_fpath=None):
        """Deduplicate embeddings using numpy.

        Args:
            embeddings(np.array): Raw embeddings.
            source_fpath(str): s3 path to parquet files containing index and inverse arrays.
            dest_fpath(str): s3 path to save index and inverse arrays after deduplication (these will be saved to .parquet).

        Returns:
            np.array: Deduplicated embeddings

        """

        if source_fpath is not None:
            index = wr.s3.read_parquet(f"{source_fpath}index.parquet")
            self._dedupe_index = np.array(index["index"].tolist())
            inverse = wr.s3.read_parquet(f"{source_fpath}inverse.parquet")
            self._dedupe_inverse = np.array(inverse["inverse"].tolist())
            embeddings = embeddings[index]
        else:
            embeddings, self._dedupe_index, self._dedupe_inverse = np.unique(
                embeddings, return_index=True, return_inverse=True, axis=0
            )

        print(f"\nNumber of unique embeddings: {embeddings.shape[0]}")

        if dest_fpath is not None:
            index_df = pd.DataFrame({"index": list(self._dedupe_index)})
            inverse_df = pd.DataFrame({"inverse": list(self._dedupe_inverse)})
            wr.s3.to_parquet(df=index_df, path=f"{dest_fpath}index.parquet")
            wr.s3.to_parquet(df=inverse_df, path=f"{dest_fpath}inverse.parquet")

        return embeddings

    def faiss_knn(self, embeddings_train, embeddings_search, k=10, batch_size=1000):
        """Calculate k-nearest neighbours using Faiss.

        Args:
            embeddings_train(np.array): Embeddings to train/add to Faiss index.
            embeddings_search(np.array): Query embeddings to search Faiss index with (can be the same as embeddings_train).
            k(int): Number of neighbours.
            batch_size(int): Batch size for nearest neighbours search.

        Returns:
            np.array: kNN indices.
            np.array: kNN distances.
        """
        print("\n Calculating KNN using FAISS...")
        index_ = faiss.IndexFlatL2(embeddings_train.shape[1])

        try:
            ngpus = faiss.get_num_gpus()
            print(f"\n Moving index to {ngpus} GPU(s)...")
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index_, co)

        except:
            nlist = 100
            index = faiss.IndexIVFFlat(index_, embeddings_train.shape[1], nlist)
            index.nprobe = 10

        print("\n Training FAISS index...")
        index.train(embeddings_train.astype(np.float32))
        print("\n Adding data to index...")
        index.add(embeddings_train.astype(np.float32))
        print("\n Searching index...")

        distances = []
        indices = []
        for i in tqdm(range(0, embeddings_search.shape[0], batch_size)):
            dist, ind = index.search(
                embeddings_search[i : i + batch_size].astype(np.float32), k=k
            )
            distances.append(dist)
            indices.append(ind)

        print("\n Finished calculating KNN.")
        distances = np.concatenate(distances, axis=0)
        indices = np.concatenate(indices, axis=0)

        return indices, distances

    def save_knn(self, indices, distances):
        """Save k-nearest neighbour index and distance arrays by appending them to parquet files.

        Args:
            indices(np.array): kNN indices.
            distances(np.array): kNN distances.

        """
        if self._dedupe_inverse is not None:
            self.data["knn_ind"] = list(indices[self._dedupe_inverse])
            self.data["knn_dist"] = list(distances[self._dedupe_inverse])
        else:
            self.data["knn_ind"] = list(indices)
            self.data["knn_dist"] = list(distances)

        print("\n Appending knn results to parqet...")
        self.save(colname="knn_ind")
        self.save(colname="knn_dist")
