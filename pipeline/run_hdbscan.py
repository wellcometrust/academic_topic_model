import click

from pipeline.clustering.clustering import HDBSCANClusters


@click.command()
@click.option("--min_cluster_size", default="50")
@click.option("--min_samples", default="50")
@click.option("--epsilon", default="0.0")
@click.option("--n_params", default=None)
@click.option("--sample_frac", default=1.0)
@click.option("--crop_coordinates", default="-20,20,-20,20")
@click.option("--embedding_col", default="umap_5dim_15n")
@click.option("--wandb", is_flag=True)
@click.option(
    "--input_s3_uri",
    "-i",
    default="s3://datalabs-data/funding_impact_measures/embeddings_sample/embeddings.parquet",
)
def calculate_clusters(
    min_cluster_size,
    min_samples,
    epsilon,
    n_params,
    sample_frac,
    crop_coordinates,
    embedding_col,
    wandb,
    input_s3_uri,
):
    input_s3_paths = [fpath.strip() for fpath in input_s3_uri.split(",")]

    min_cluster_size = [int(cs.strip()) for cs in min_cluster_size.split(",")]
    epsilon = [float(e.strip()) for e in epsilon.split(",")]
    if isinstance(min_samples, str):
        min_samples = [int(s.strip()) for s in min_samples.split(",")]
    else:
        min_samples = [min_samples]

    crop_coordinates = [int(c.strip()) for c in crop_coordinates.split(",")]

    clustering_pipeline = HDBSCANClusters(
        input_bucket=input_s3_paths,
        embedding_col=embedding_col,
        crop_coordinates=crop_coordinates,
    )

    if n_params is not None:
        print("\nRunning parameter search...")
        clustering_pipeline.run_param_search(
            params={
                "min_samples": min_samples,
                "min_cluster_size": min_cluster_size,
                "cluster_selection_epsilon": epsilon,
            },
            n=int(n_params),
            sample_frac=float(sample_frac),
            log_to_wandb=wandb,
        )
    else:
        clustering_pipeline.run_hdbscan(
            min_cluster_size=min_cluster_size[0],
            min_samples=min_samples[0],
            cluster_selection_epsilon=epsilon[0],
        )
        clustering_pipeline.save_clusters(mcs=min_cluster_size[0], ms=min_samples[0])


if __name__ == "__main__":
    calculate_clusters()
