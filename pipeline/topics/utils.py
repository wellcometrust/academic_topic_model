class EmptyDimensionalityReduction:
    """Empty dimensionality reduction model to use with pre-calculated embeddings."""

    def __init__(self, umap_embeddings):
        self.umap_embeddings = umap_embeddings
        """ Initialise model with existing embeddings."""

    def fit(self, X):
        return self

    def transform(self, X):
        return self.umap_embeddings


class EmptyClusterModel:
    """Empty cluster model to use with pre-calculated clusters."""

    def __init__(self, cluster_results):
        self.cluster_results = cluster_results
        """ Initialise model with existing cluster IDs."""

    def fit(self, X):
        self.labels_ = self.cluster_results
        return self

    def predict(self, X):
        return X
