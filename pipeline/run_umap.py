import click
from clustering.clustering import UMAPEmbeddings


@click.command()
@click.option("--embedding_dim", "-d", default=2)
@click.option("--n_neighbours", default=15)
@click.option("--metric", default="euclidean")
@click.option("--min_dist", default=0.0)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--save_plot", is_flag=True)
@click.option(
    "--input_s3_uri",
    "-i",
    default="s3://datalabs-data/funding_impact_measures/embeddings_sample/"
    "embeddings.parquet",
)
@click.option("--precalculate_knn", is_flag=True)
@click.option("--save_knn", is_flag=True)
@click.option("--dedupe_embeddings", is_flag=True)
@click.option("--deduped_embeddings_fpath", default=None)
def calculate_umap(
    embedding_dim,
    n_neighbours,
    metric,
    min_dist,
    verbose,
    save_plot,
    input_s3_uri,
    precalculate_knn,
    save_knn,
    dedupe_embeddings,
    deduped_embeddings_fpath,
):
    input_s3_paths = [fpath.strip() for fpath in input_s3_uri.split(",")]

    embedding_pipeline = UMAPEmbeddings(input_bucket=input_s3_paths)

    print(f"\nTotal number of embeddings: {len(embedding_pipeline.data)}")

    embedding_pipeline.run_umap(
        n_components=embedding_dim,
        n_neighbors=n_neighbours,
        metric=metric,
        min_dist=min_dist,
        verbose=verbose,
        precalculate_knn=precalculate_knn,
        save_knn=save_knn,
        dedupe=dedupe_embeddings,
        path_to_deduped_files=deduped_embeddings_fpath,
    )

    embedding_pipeline.save_embeddings(
        dim=embedding_dim, n=n_neighbours
    )

    if save_plot:
        embedding_pipeline.plot_embeddings(alpha=0.1, s=0.1)


if __name__ == "__main__":
    calculate_umap()
