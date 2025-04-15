import os
import random

import awswrangler as wr
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm

from pipeline.topics.utils import EmptyClusterModel
from pipeline.topics.utils import EmptyDimensionalityReduction
from pipeline.utils.llm import LLM


class TopicHierarchy(LLM):
    """Topic labelling with a LLM."""

    def __init__(self, input_bucket, topic_model, topic_info, llm_id="mixtral"):
        """Initialise parent class and parameters.

        Args:
            input_bucket(list, str): S3 URI or list of S3 URIs for text embedding directory.
            topic_model(str, BERTopic): (Path to) BERTopic model.
            topic_info(str, pd.DataFrame): (Path to) topic metadata.
            llm_id(str): ID/Name of LLM.

        """
        super().__init__(llm_id)

        self.abstracts = self.load_abstracts(input_bucket)
        self.system_message = self.get_system_message()

        if isinstance(topic_model, str) and os.path.exists(topic_model):
            topic_model = BERTopic.load(topic_model)
        self.topic_model = topic_model

        if isinstance(topic_info, str) and os.path.exists(topic_info):
            self.topic_info = pd.read_csv(topic_info)
        else:
            self.topic_info = topic_info
        self.topic_info["Topic"] = self.topic_info["Topic"].astype(str)
        self.topic_names = self.topic_info.set_index("Topic")["llm_topic"].to_dict()

    def load_abstracts(self, input_bucket):
        """Load publication abstracts from parquet.

        Args:
            input_bucket(list, str): S3 URI or list of S3 URIs for text embedding directory.

        Returns:
            list: Abstract texts.
        """
        print("\nLoading document abstracts...")
        if not isinstance(input_bucket, list):
            input_bucket = [input_bucket]
        fpaths = [fpath for ib in input_bucket for fpath in wr.s3.list_objects(ib)]
        abstracts = wr.s3.read_parquet(fpaths, columns=["abstract"])
        abstracts = abstracts.dropna()
        return abstracts["abstract"].tolist()

    def get_hierarchy(self, **kwargs):
        """Calculate topic hierarchy.

        Args:
            **kwargs: BERTopic parameters for hierarchical topic calculation.

        Returns:
            pd.DataFrame: Hierarchical topics and distances.
        """
        print("Calculating topic hierarchy...")
        return self.topic_model.hierarchical_topics(self.abstracts, **kwargs)

    def get_system_message(self):
        """Main body of the LLM prompt containing subtopics.

        Returns:
            str: Main prompt containing subtopics.
        """
        return """
        Task: Generate a short parent-level topic label based on the provided subtopics.
        Instructions:
        Use only the provided subtopics to generate a topic label.
        Do not include any explanations or apologies in your response.
        Do not include any text except the generated topic label.
        Make sure the topic label is not longer than 10 words.

        Example:
        - Art History, Theory and Criticism
        - Music
        - Performing Arts
        - Screen and Digital Media
        - Visual Arts
        Topic: Creative Arts

        Here are the subtopics I want you to consider:
        {subtopics}
        Topic:
        """

    def format_topics(self, topic_ids):
        """Format topic names into a bulleted list to be used as part of the LLM prompt.

        Args:
            topic_ids(list): List of subtopic IDs.

        Returns:
            str: Bulleted list of topic names.
        """
        topic_names = [self.topic_names[str(t)] for t in topic_ids]
        return "\n".join([f"- {t}" for t in topic_names])

    def llm_prompt(self, topic_ids, max_topics):
        """Main body of the LLM prompt containing list of actual subtopics.

        Args:
            topic_ids(list): List of subtopic IDs.
            max_topics(int): If the number of topic IDs is larger than max_topics, a sample of
                size max_topics will be passed to the LLM.

        Returns:
            str: Main prompt containing subtopics to label.
        """
        if len(topic_ids) > max_topics:
            topic_ids = random.sample(topic_ids, max_topics)
        return self.system_message.format(subtopics=self.format_topics(topic_ids))

    def dataloader(self, hierarchy, max_topics):
        """Dataloader for LLM prompts.

        Args:
            hierarchy(pd.DataFrame): Hierarchical topics and distances.
            max_topics(int): If the number of topic IDs is larger than max_topics, a sample of
                size max_topics will be passed to the LLM.

        Yields:
            str: LLM prompt.
        """
        for topic_ids in hierarchy["Topics"].tolist():
            llm_prompt = self.llm_prompt(topic_ids, max_topics)
            yield llm_prompt

    def get_llm_topics(self, hierarchy, max_topics=100):
        """Generate hierarchical topic labels using an LLM.

        Args:
            hierarchy(pd.DataFrame): Hierarchical topics and distances.
            max_topics(int): If the number of topic IDs is larger than max_topics, a sample of
                size max_topics will be passed to the LLM.

        Returns:
            list: Hierarchical topic labels generated by the LLM.
        """
        print("Getting new topic labels from LLM...")
        topic_labels = []
        for prompt in tqdm(
            self.dataloader(hierarchy, max_topics), total=len(hierarchy)
        ):
            topic_labels.append(self.run(prompt, retry=True))
        return topic_labels

    def update_hierarchy_labels(self, hierarchy, max_topics=100):
        """Update hierarchical topic metadata with LLM-generated labels.

        Args:
            hierarchy(pd.DataFrame): Hierarchical topics and distances.
            max_topics(int): If the number of topic IDs is larger than max_topics, a sample of
                size max_topics will be passed to the LLM.

        Returns:
            pd.DataFrame: Hierarchical topic metadata table with LLM-generated labels.
        """
        llm_labels = self.get_llm_topics(hierarchy, max_topics)
        hierarchy["Parent_Name"] = llm_labels
        self.topic_names.update(
            hierarchy.set_index("Parent_ID")["Parent_Name"].to_dict()
        )
        hierarchy["Child_Left_Name"] = hierarchy["Child_Left_ID"].map(self.topic_names)
        hierarchy["Child_Right_Name"] = hierarchy["Child_Right_ID"].map(
            self.topic_names
        )
        return hierarchy

    def update_topic_info(self, hierarchy, threshold=0.5, levels=3):
        """Update topic metadata with LLM-generated parent labels.

        Args:
            hierarchy(pd.DataFrame): Hierarchical topics with distances and LLM labels.
            threshold(float): Minimum distance to determine parent topics.
            levels(int): How many levels of parent topics to assign (parent, parent_parent, etc.)
        """
        hierarchy = hierarchy.explode("Topics")
        hierarchy["Topics"] = hierarchy["Topics"].astype(str)
        hierarchy.sort_values("Distance", inplace=True)
        hierarchy = (
            hierarchy[hierarchy["Distance"] >= threshold]
            .groupby("Topics")["Parent_Name"]
            .apply(list)
        )
        self.topic_info = self.topic_info.merge(
            hierarchy, left_on="Topic", right_index=True
        )
        for l in range(levels):
            self.topic_info[f"parent{'_parent'*l}"] = self.topic_info[
                "Parent_Name"
            ].apply(lambda ls: ls[l])
        self.topic_info.drop("Parent_Name", axis=1, inplace=True)

    @staticmethod
    def _truncate_labels(original_labels):
        """Truncate very long topic labels (>100 characters) for visualisation.

        Args:
            original_labels(list): List of topic labels.

        Returns:
            list: Truncated topic labels.
        """
        custom_labels = []
        for t in original_labels:
            if len(t) > 100:
                cl = f"{t[:100]}..."
                custom_labels.append(cl)
            else:
                custom_labels.append(t)
        return custom_labels

    def visualise_hierarchy(self, hierarchy, output_fpath, **kwargs):
        """Interactive visualisation of topic hierarchy with LLM-generated topic labels.

        Args:
            hierarchy(pd.DataFrame): Hierarchical topics with distances and LLM labels.
            output_fpath(str): Path to save HTML output to.
            **kwargs: Additional arguments passed to BERTopic hierarchical visualisation.

        Returns:
            plotly.graph_objects.Figure: Interactive visualisation of topic hierarchy.
        """
        new_topic_labels = self._truncate_labels(self.topic_info["llm_topic"].tolist())
        new_topic_labels = [
            f"{t} ({c:,})"
            for t, c in zip(new_topic_labels, self.topic_info["Count"].tolist())
        ]
        self.topic_model.set_topic_labels(new_topic_labels)
        vis = self.topic_model.visualize_hierarchy(
            hierarchical_topics=hierarchy, custom_labels=True, **kwargs
        )
        vis.write_html(output_fpath)
        return vis
