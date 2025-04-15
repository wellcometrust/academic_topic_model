import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.model_selection import ParameterGrid
from umap import UMAP
from pipeline.utils.base import BaseModel
from pipeline.utils.knn import KNN

try:
    import wandb
except ModuleNotFoundError:
    print("Unable to log parameter search to Weights & Biases."
          " Make sure to install wandb to enable this functionality.")


class UMAPEmbeddings(KNN):
    """Methods to perform dimensionality reduction on embeddings using UMAP."""

    def __init__(self, input_bucket, embedding_col="embeddings"):
        additional_cols = ["knn_ind", "knn_dist"]

        super().__init__(input_bucket, embedding_col, additional_cols=additional_cols)
        """ Initialise parent class and parameters.

        Args:
            input_bucket(list, str): S3 URI or list of S3 URIs for text
                embedding directory.
            embedding_col(str): Name of column containing text embeddings.

        """

    def run_umap(
        self,
        precalculate_knn=False,
        save_knn=False,
        n_neighbors=15,
        verbose=True,
        dedupe=True,
        path_to_deduped_files=None,
        **umap_kwargs,
    ):
        """Calculate UMAP embeddings (on GPU, if available).

        Args:
            precalculate_knn(bool): Whether to precalculate k-nearest neighbours with Faiss.
            save_knn(bool): Whether to save precalculated k-nearest neighbours.
            n_neighbors(int): Number of neighbours.
            verbose(bool): Whether to run UMAP in verbose mode.
            dedupe(bool): Whether to deduplicate embeddings using numpy.
            path_to_deduped_files(str): s3 path to parquet files containing deduplicated indices.
            **umap_kwargs: UMAP parameters.
        """
        print("\n Calculating UMAP embeddings...")
        embeddings = np.array(self.data[self._embedding_col].tolist(), dtype=np.double)

        if dedupe:
            self.data.drop("embeddings", axis=1, inplace=True)
            print("\nDeduplicating embeddings...")
            embeddings = self.dedupe_embeddings(
                embeddings=embeddings, source_fpath=path_to_deduped_files
            )

        if precalculate_knn:
            knn = self.faiss_knn(
                embeddings_train=embeddings, embeddings_search=embeddings, k=n_neighbors
            )
            umap_kwargs["precomputed_knn"] = knn
            if save_knn:
                self.save_knn(*knn)
        elif all([k in self.data.columns for k in ["knn_ind", "knn_dist"]]):
            print("\nUsing knn from file...")
            knn_ind = np.array(self.data["knn_ind"].tolist())[self._dedupe_index]
            knn_dist = np.array(self.data["knn_dist"].tolist())[self._dedupe_index]
            umap_kwargs["precomputed_knn"] = (knn_ind, knn_dist)

        umap_embeddings = UMAP(verbose=verbose, **umap_kwargs).fit_transform(embeddings)

        self.data["umap"] = list(umap_embeddings[self._dedupe_inverse])

        print("\nFinished calculating UMAP embeddings.")

    def plot_embeddings(self, save=True, **kwargs):
        """Create a scatterplot of 2-dimensional UMAP embeddings.

        Args:
            save(bool): If true, saves plot as umap_embeddings.png.
            **kwargs: Matplotlib parameters.
        """
        umap_embeddings = np.array(self.data["umap"].tolist())
        if umap_embeddings.shape[1] > 2:
            raise ValueError("Unable to plot >2-dimensional embeddings.")

        plt.figure(figsize=(15, 15))
        plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], **kwargs)
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        if save:
            plt.savefig("umap_embeddings.png")

    def save_embeddings(self, **kwargs):
        """Save UMAP embeddings by attaching them as a column to original parquet files.

        Args:
            **kwargs: Keyword arguments to attach to the column name.
        """
        if "umap" not in self.data.columns:
            raise ValueError(
                "UMAP embeddings cannot be None. Make sure to run run_umap"
                " before trying to save embeddings."
            )

        print("\n Appending UMAP embeddings to parquet...")
        self.save(colname="umap", **kwargs)


class HDBSCANClusters(BaseModel):
    """Methods to perform HDBSCAN clustering on embeddings."""

    def __init__(self, input_bucket, embedding_col="umap", crop_coordinates=None):
        additional_cols = None
        if crop_coordinates is not None:
            additional_cols = [embedding_col.replace("5dim", "2dim")]

        super().__init__(
            input_bucket=input_bucket,
            embedding_col=embedding_col,
            additional_cols=additional_cols,
        )
        """ Initialise parent class and parameters.

        Args:
            input_bucket(list, str): S3 URI or list of S3 URIs for text embedding directory.
            embedding_col(str): Name of column containing UMAP embeddings.

        """
        if crop_coordinates is not None:
            if len(crop_coordinates) == 4:
                self._crop_data(crop_coordinates)
            else:
                raise ValueError("XY coordinates for cropping must have a length of 4.")

    def _crop_data(self, coordinates):
        """
        Coordinates in format (x_min, x_max, y_min, y_max)
        """
        embedding_col_2d = self._embedding_col.replace("5dim", "2dim")
        embeddings_2d = np.array(self.data[embedding_col_2d].tolist())
        self.data["x"] = embeddings_2d[:, 0]
        self.data["y"] = embeddings_2d[:, 1]
        self.data = self.data[
            (self.data["x"] > coordinates[0])
            & (self.data["x"] < coordinates[1])
            & (self.data["y"] > coordinates[2])
            & (self.data["y"] < coordinates[3])
        ]

    def run_hdbscan(self, **kwargs):
        """Calculate clusters using HDBSCAN (on GPU, if available).

        Args:
            **kwargs: HDBSCAN parameters.
        """
        embeddings = np.array(self.data[self._embedding_col].tolist())

        print("\n Calculating HDBSCAN clusters...")

        hdbscan = HDBSCAN(**kwargs).fit(embeddings)
        self.data["hdbscan_labels"] = hdbscan.labels_

    def _get_random_params(self, params, n, random_seed):
        """Randomly sample from parameter grid.

        Args:
            params(dict): HDBSCAN parameters.
            n(int): Sample size.
            random_seed(int): Random seed.

        Returns:
            list: Random parameter sample.
        """
        random.seed(random_seed)
        param_grid = ParameterGrid(params)
        all_params = []
        for p in param_grid:
            all_params.append(p)
        return random.sample(all_params, n)

    @staticmethod
    def _get_metrics(hdbscan):
        """Calculate HDBSCAN metrics.

        Args:
            hdbscan(hdbscan.HDBSCAN): HDBSCAN object.

        Returns:
            dict: Validity, topic count, and coverage metrics.
        """
        metrics = {
            "validity": hdbscan.relative_validity_,
            "topic_count": hdbscan.labels_.max() + 1,
            "coverage": sum(hdbscan.labels_ != -1) / len(hdbscan.labels_),
        }
        return metrics

    def _wandb_param_search(self, embeddings, config=None):
        """Run a sweep with weights and biases.

        Args:
            embeddings(np.array): Embeddings to cluster.
            config(dict): Sweep configuration.
        """
        with wandb.init(config=config):
            config = wandb.config
            min_samples = wandb.config.min_samples
            min_cluster_size = wandb.config.min_cluster_size
            cluster_selection_epsilon = wandb.config.cluster_selection_epsilon
            hdbscan = HDBSCAN(
                gen_min_span_tree=True,
                min_samples=min_samples,
                min_cluster_size=min_cluster_size,
                cluster_selection_epsilon=cluster_selection_epsilon,
            ).fit(embeddings)
            wandb.log(self._get_metrics(hdbscan))

    def _random_param_search(self, embeddings, config, n, random_seed=123):
        """Conduct a random search of HDBSCAN parameters.

        Args:
            embeddings(np.array): Embeddings to cluster.
            config(dict): HDBSCAN parameters to include in the random search.
            n(int): Sample size.
            random_seed(int): Random seed.
        """
        results = []
        for i, random_params in enumerate(
            self._get_random_params(
                params=dict(
                    min_samples=config["parameters"]["min_samples"]["values"],
                    min_cluster_size=config["parameters"]["min_cluster_size"]["values"],
                    cluster_selection_epsilon=config["parameters"][
                        "cluster_selection_epsilon"
                    ]["values"],
                ),
                n=n,
                random_seed=random_seed,
            )
        ):
            print(f"\nRound {i}: {random_params}")
            hdbscan = HDBSCAN(gen_min_span_tree=True, **random_params).fit(embeddings)
            res = self._get_metrics(hdbscan)
            res.update(random_params)
            print(res)
            results.append(res)
        pd.DataFrame(results).to_csv("hdbscan_paramsearch.csv")

    def run_param_search(
        self,
        params,
        n,
        random_seed=123,
        sample_frac=1.0,
        log_to_wandb=True,
        method="random",
        project_name="publication-embeddings-hdbscan"
    ):
        """Conduct a HDBSCAN parameter search. Output metrics include the vaildity score, number of topics, and coverage (proportion of
        non-outlier points).

        Args:
            params(dict): HDBSCAN parameters to include in grid search.
            n(int): Sample size.
            random_seed(int): Random seed.
            sample_frac(float): Embeddings sample size.
            log_to_wandb(bool): Whether to log results to Weights & Biases.
            method(str): Weights & Biases parameter search method.
        """
        if sample_frac < 1:
            embeddings = np.array(
                self.data[self._embedding_col].sample(frac=sample_frac).tolist()
            )
            print(embeddings.shape)
        else:
            embeddings = np.array(self.data[self._embedding_col].tolist())

        parameters_dict = {
            "method": method,
            "name": "sweep",
            "metric": {"goal": "maximize", "name": "validity"},
            "parameters": {
                "dataset": {"value": self._input_bucket},
                "sample_frac": {"value": sample_frac},
                "n_documents": {"value": len(embeddings)},
            },
        }
        parameters_dict["parameters"].update(
            {k: {"values": v} for k, v in params.items()}
        )

        if log_to_wandb:
            sweep_id = wandb.sweep(
                sweep=parameters_dict, project=project_name
            )
            wandb.agent(
                sweep_id=sweep_id,
                function=partial(self._wandb_param_search, embeddings=embeddings),
                count=n,
            )
        else:
            self._random_param_search(
                embeddings=embeddings,
                config=parameters_dict,
                n=n,
                random_seed=random_seed,
            )

    def save_clusters(self, **kwargs):
        """Save HDBSCAN cluster labels by attaching them as a column to original parquet files.

        Args:
            **kwargs: Keyword arguments to attach to the column name.
        """
        if "hdbscan_labels" not in self.data.columns:
            raise ValueError(
                "HDBSCAN labels cannot be None. Make sure to run run_hdbscan before trying to save clusters."
            )

        print("\n Appending cluster labels to parquet...")
        self.save(colname="hdbscan_labels", **kwargs)
