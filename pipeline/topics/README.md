# Topics

## Topic modelling
To generate a topic model:
```
python3 pipeline/topics/topic_modelling.py --input_s3_uri="s3://path-to-your/embeddings"
```
This will generate a BERTopic topic model with c-TF-IDF topic representations using default parameters on pre-calculated embeddings and with pre-calculated cluster assignments. Topic IDs will be saved to the provided embeddings files in a new column named "topics" by default. A pickled topic model will be saved to the `--fpath` provided, which is "topic_model" unless specified. Several methods are available for subsequent topic refinement as described below.

You may wish to run outlier reduction as part of the initial topic modelling step:
```
python3 pipeline/topics/topic_modelling.py --input_s3_uri="s3://path-to-your/embeddings" --reduce_outliers
```
This will decrease the number of documents labelled as outliers (-1) by HDBSCAN using a strategy which finds the best matching topic using the top-k nearest neighbours. The new topic assignments will be saved to a column named "topics_ored" and a topic mode will be saved with the same "_ored" suffix.

Please note that topic modelling has high memory requirements - if you run this on a large dataset, you may need to choose a memory-optimised ec2 instance (e.g.: r6a.4xlarge, r6a.8xlarge).

## Topic labelling and hierarchy
We will use [Ollama](https://ollama.com/) to label topics with an LLM. Ollama is a LLM library which allows you to easily run different LLMs locally (on a GPU). It uses 4-bit quantisation by default and requires only minimal setup.
1. Install Ollama `curl -fsSL https://ollama.com/install.sh | sh` or follow the instructions [here](https://ollama.com/download/linux) for other operating systems.
2. Find a suitable model in the [library](https://ollama.com/library).
3. Pull the model(s) you want to use by running `ollama pull model_name`.

You need a GPU to label topics with the LLM, such as a g5 instance.

You may want to run topic labelling from within a Jupyter notebook to allow for easier inspection of the results:
```python
from pipeline.topics.topic_labelling import LLMLabels

# LLM labelling model
llm_labelling_model = LLMLabels(
        input_bucket="s3://datalabs-data/funding_impact_measures/embeddings_sample/", # s3 path to your embeddings
        topic_model="topic_model" # path to your topic model
    )

# Run topic label generation
llm_labels = llm_labelling_model.get_llm_topics()

# Extract topic labels
llm_labels_clean = llm_labelling_model.clean_llm_topics(llm_labels)

# Save topic info containing LLM labels
llm_labelling_model.save_llm_topics(llm_labels_clean, dest_fpath="topic_info_llm.csv")
```

You may wish to run several rounds of topic label generation with different random seeds, which will pass different samples of document titles to the LLM:
```
llm_labels = llm_labelling_model.get_llm_topics(random_state=456)
```
You can then select the best or majority label for each topic.

To generate a topic hierarchy and label parent topics with Llama:
```python
from pipeline.topics.topic_hierarchy import TopicHierarchy

# Hierarchical LLM topic labelling model
llm_labelling_model = TopicHierarchy(
    input_bucket="s3://datalabs-data/funding_impact_measures/embeddings_sample/", # s3 path to your embeddings
    topic_model="topic_model", # path to your topic model
    topic_info="topic_info_llm.csv" # path to your topic labels/info
    )

# Calculate hierarchy using BERTopic hierarchical topic modelling
hierarchy = llm_labelling_model.get_hierarchy()

# Generate parent topic labels
llm_hierarchy = llm_labelling_model.update_hierarchy_labels(hierarchy)

# Save hierarchy and labels
llm_hierarchy.to_csv("topic_hierarchy_llm.csv")
```

## Topic refinement
The best topic modelling results are often achieved after some manual refinement, which is ideally carried out after the topic labelling and hierarchy calculation steps. You can find real examples of topic refinement in the notebooks folder [here](../../notebooks/funder_comparison/) or [here](../../notebooks/discovery_research/).

It is recommended to visualise the topics and hierarchy before each refinement step. There are several options to do this:

**Visualise topics per cluster**

To generate an interactive cluster visualisation with document titles displayed on hover:
```python
import pandas as pd
from viz.clusters.visualise_documents import visualise_documents

# Load relevant data from embeddings files
data = wr.s3.read_parquet(
    "s3://datalabs-data/funding_impact_measures/embeddings_sample/",
    columns=["id", "title", "umap_2dim_15n", "topics"]
    )

# Load LLM-generated topic labels and add to dataset
topic_info = pd.read_csv("topic_info_llm.csv")
data = data.merge(topic_info, how="left", left_on="topics", right_on="Topic")

# Generate interactive Plotly visualisation
vis = visualise_documents(docs=data["title"].tolist(),
                          topics=data["Topic"].tolist(),
                          topic_names=data["llm_topic"].tolist(),
                          reduced_embeddings=np.array(data["umap_2dim_15n"].tolist())
                          )
vis.write_html("clusters.html")
```

**Visualise hierarchy**

To print a simple hierarchical tree with topic IDs:

```python
tree = llm_labelling_model.topic_model.get_topic_tree(llm_hierarchy)
print(tree)
.
└─Cognitive Neuroscience
     ├─Artificial Intelligence and Neuroscience
     │    ├─■──Psychological Theories ── Topic: 21
     │    └─■──Philosophy and Ethics ── Topic: 124
     └─■──Animal Behavior and Cognition ── Topic: 29
```

To generate an interactive hierarchy using Plotly:
```python
vis = llm_labelling_model.visualise_hierarchy(
    llm_hierarchy,
    output_fpath="topic_hierarchy.html",
    title='<b>Topic hierarchy</b>',
    width=1600
    )
```

### Topic refinement options
There are several ways in which topics can be refined:

**Split a large cluster into several smaller clusters**
```python
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from pipeline.topics.topic_modelling import TopicModel

# 1. Initialise a new topic model with the ID of the cluster you want to split
topic_model_cluster0 = TopicModel(
    input_bucket="s3://datalabs-data/funding_impact_measures/embeddings_sample/", # s3 path to your embeddings
    umap_embedding_col="", # no need to load UMAP embeddings
    cluster_col="topics_ored", # column containing topic/cluster IDs
    cluster_id=0 # ID of the cluster you want to split
)

# 2. Initialise component models
vectorizer_model = CountVectorizer(stop_words="english")
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
umap_model = UMAP() # specify additional UMAP parameters as needed
hdbscan_model = HDBSCAN() # specify additional HDBSCAN parameters as needed

# 3. Run topic modelling
topic_model_cluster0.run_bertopic(dimensionality_model=umap_model, cluster_model=hdbscan_model, vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model)

# 4. Check out the new topics (e.g. with topic_model_cluster0.topic_model.get_topic_info())

# 4. Run outlier reduction (optional)
topic_model_cluster0.reduce_outliers(update_topics=True)

# 5. Increment topic IDs to avoid overlap with original
topic_model_cluster0.data['topics_ored'] = topic_model_cluster0.data["topics_ored"] + 1000

# 6. Create New topic ID mapping
cluster_id_dict = topic_model_cluster0.data['topics_ored'].to_dict()

# Steps 1-6 can be repeated for other large clusters you want to split

# Update full topic model with new clusters
topic_model = TopicModel(
    input_bucket="s3://datalabs-data/funding_impact_measures/embeddings_sample/", # s3 path to your embeddings
    umap_embedding_col="umap_5dim_15n", # UMAP embedding column
    cluster_col="topics_ored", # cluster/topic ID column
)

# Add in new cluster IDs
topic_model.update_cluster_ids(cluster_id_dict)

# Update topic model by re-running BERTopic
topic_model.run_bertopic()

# Save new topic model
topic_model.save_topic_model("topic_model_refined")
```

**Remove topics (set as outlier)**
```python
from pipeline.topics.topic_modelling import TopicModel

# Create a list of topic IDs you want to remove
outlier = [305, 360, 376]

# Initialise topic model
topic_model = TopicModel(
    input_bucket="s3://datalabs-data/funding_impact_measures/embeddings_sample/", # s3 path to your embeddings
    umap_embedding_col="umap_5dim_15n", # UMAP embedding column
    cluster_col="topics_ored", # cluster/topic ID column
    topic_model="topic_model_ored" # path to topic model you want to refine
)

# Update topic model
topic_model.to_outlier(outlier)

# Save new topic model
topic_model.save_topic_model("topic_model_refined")
```

**Merge topics**
```python
from pipeline.topics.topic_modelling import TopicModel

# Create a list of lists of topic IDs you want to merge
to_merge = [
    [233,12,5], # topics 233, 12, and 5 will be merged into a single topic
    [265,390,385]
]

# Initialise topic model
topic_model = TopicModel(
    input_bucket="s3://datalabs-data/funding_impact_measures/embeddings_sample/", # s3 path to your embeddings
    umap_embedding_col="umap_5dim_15n", # UMAP embedding column
    cluster_col="topics_ored", # cluster/topic ID column
    topic_model="topic_model_ored" # path to topic model you want to refine
)

# Merge topics
topic_model.merge_topics(to_merge)

# Save new topic model
topic_model.save_topic_model("topic_model_refined")
```
