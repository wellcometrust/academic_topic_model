import os

import awswrangler as wr
import click
from bertopic import BERTopic
from tqdm import tqdm

from pipeline.topics.utils import EmptyClusterModel
from pipeline.topics.utils import EmptyDimensionalityReduction
from pipeline.utils.llm import LLM


class LLMLabels(LLM):
    """Methods to perform topic labelling with a LLM."""

    def __init__(self, input_bucket, topic_model, llm_id="mixtral"):
        """Initialise parent class and parameters.

        Args:
            input_bucket(list, str): S3 URI or list of S3 URIs for text embedding directory.
            topic_model(str, BERTopic): (Path to) BERTopic model.
            llm_id(str): ID/Name of LLM.

        """
        super().__init__(llm_id)

        self.system_message = self.get_system_message()
        self.titles = self.load_titles(input_bucket)
        if isinstance(topic_model, str) and os.path.exists(topic_model):
            self.topic_model = BERTopic.load(topic_model)
        elif isinstance(topic_model, BERTopic):
            self.topic_model = topic_model
        else:
            raise TypeError("Topic model must be a (path to a) BERTopic model.")

        self._sample_titles = None

    def load_titles(self, input_bucket, id_col="publication_id"):
        """Load publication titles from parquet.

        Args:
            input_bucket(list, str): S3 URI or list of S3 URIs for text embedding directory.
            id_col(str): Name of column containing publication IDs.

        Returns:
            pd.DataFrame: Title texts.
        """
        print("\nLoading document titles...")
        if not isinstance(input_bucket, list):
            input_bucket = [input_bucket]
        fpaths = [fpath for ib in input_bucket for fpath in wr.s3.list_objects(ib)]
        titles = wr.s3.read_parquet(fpaths, columns=[id_col, "title"])
        titles = titles.dropna()
        return titles

    @staticmethod
    def format_titles(title_ls, max_len=200):
        """Truncate very long titles (>max_len) and format into a bulleted list to be used as part of the LLM prompt.

        Args:
            title_ls(list): List of publication titles.

        Returns:
            str: Bulleted list of publication titles.
        """
        ls = []
        for t in title_ls:
            if len(t) > max_len:
                ls.append(f"- {t[:max_len]}...")
            else:
                ls.append(f"- {t}")
        return "\n".join(ls)

    def get_system_message(self):
        """Main body of the LLM prompt containing keywords and list of document titles.

        Returns:
            str: Main prompt containing keywords and titles.
        """
        return """
        Task: Generate a short topic label based on the provided keywords and a small sample of documents.
        Instructions:
        Use only the provided keywords and document sample to generate a topic label.
        Consider all the provided information to create a label. Do not summarise individual documents.
        Do not include any explanations or apologies in your responses.
        Do not include any text except the generated topic label.
        Make sure the topic label is not longer than 10 words.

        Example:
        Keywords: connectivity, cortex, task, cortical, visual, memory, frontal, gyrus, fmri, neural
        Document sample:
        - Modelling the Effects of Ongoing Alpha Activity on Visual Perception: The Oscillation-Based Probability of Response
        - Auditory activity is diverse and widespread throughout the central brain of Drosophila
        - Categorical interoception: the role of disease context and individual differences in habitual symptom reporting
        - Imaging Markers for the Characterization of Gray and White Matter Changes from Acute to Chronic Stages after Experimental Traumatic Brain Injury
        - Chapter 7 Multisensory and sensorimotor maps
        Topic: Neural and cognitive processes

        Here are the keywords and documents I want you to consider:
        Keywords: {keywords}
        Document sample:
        {titles}
        Topic:
        """

    def get_sample_titles(self, n, random_state):
        """Generate a sample of titles for each topic.

        Args:
            n(int): Sample size.
            random_state(int): Seed for random sampling.
        """
        sample_titles = {}
        for t, g in self.titles.groupby("topic"):
            s = g.sample(min(n, len(g)), random_state=random_state)
            sample_titles[t] = s["title"].tolist()
        self._sample_titles = sample_titles

    def dataloader(self, n_titles, random_state):
        """Dataloader for LLM prompts.

        Args:
            n_titles(int): Sample size (number of titles to pass to LLM).
            random_state(int): Seed for random sampling.

        Yields:
            str: LLM prompt.
        """
        topics = self.topic_model.get_topic_info()
        self.titles["topic"] = self.topic_model.topics_
        self.get_sample_titles(n=n_titles, random_state=random_state)
        for i in range(len(topics)):
            llm_prompt = self.system_message.format(
                titles=self.format_titles(self._sample_titles[topics.loc[i, "Topic"]]),
                keywords=topics.loc[i, "Representation"],
            )
            yield llm_prompt

    def get_llm_topics(self, n_titles=30, random_state=123):
        """Generate topic labels using an LLM.

        Args:
            n_titles(int): Sample size (number of titles to pass to LLM).
            random_state(int): Seed for random sampling.

        Returns:
            list: LLM-generated topic labels.
        """
        print("\nGenerating topic labels...")
        topic_labels = []
        for prompt in tqdm(
            self.dataloader(n_titles=n_titles, random_state=random_state),
            total=len(self.topic_model.get_topic_info()),
        ):
            topic_labels.append(self.run(prompt, retry=True))
        return topic_labels

    def save_llm_topics(self, llm_topics, dest_fpath="topic_info_llm.csv"):
        """Save topic metadata containing LLM-generated labels.

        Args:
            llm_topics(list): LLM-generated topic labels.
            dest_fpath(str): Filename/path to save topic metadata to.
        """
        topics = self.topic_model.get_topic_info()
        topics["llm_topic"] = llm_topics
        topics["titles_sample"] = topics["Topic"].map(self._sample_titles)
        topics[
            ["Topic", "Count", "Name", "Representation", "titles_sample", "llm_topic"]
        ].to_csv(dest_fpath)


@click.command()
@click.option("--llm_id", default="mixtral")
@click.option("--topic_model_path", default="data/topic_model")
@click.option(
    "--input_s3_uri",
    "-i",
    default="s3://datalabs-data/funding_impact_measures/embeddings_sample/embeddings.parquet",
)
def topic_labelling(llm_id, topic_model_path, input_s3_uri):
    input_s3_paths = [fpath.strip() for fpath in input_s3_uri.split(",")]

    llm_labelling_model = LLMLabels(
        input_bucket=input_s3_paths, topic_model=topic_model_path, llm_id=llm_id
    )

    llm_labels = llm_labelling_model.get_llm_topics()
    llm_labels = llm_labelling_model.clean_llm_topics(llm_labels)
    llm_labelling_model.save_llm_topics(llm_labels)


if __name__ == "__main__":
    topic_labelling()
