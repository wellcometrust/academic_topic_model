from langchain_community.llms import Ollama
from pipeline.topics.utils import EmptyClusterModel, EmptyDimensionalityReduction


class LLM:
    """Methods to load a LLM for inference using Ollama."""

    def __init__(self, llm_id="mixtral"):
        """Initialise LLM.

        Args:
            llm_id(str): Name/ID of LLM to load from Ollama.
        """
        self._llm_id = llm_id
        self.llm = Ollama(model=llm_id)
        self._retry_prompt = self._get_retry_prompt()

    @staticmethod
    def _get_retry_prompt():
        retry_prompt = """{prompt}

                The generated topic label is too long: {response}
                Give me an improved topic label without any explanations or apologies.
                """
        return retry_prompt

    def invoke(self, prompt):
        """Invoke a response from the LLM for the given prompt.
        If the LLM is Llama-3, "<|eot_id|>" is used as a stop token.

        Args:
            prompt(str): LLM prompt.

        Returns:
            str: LLM response.
        """
        if self._llm_id == "llama3":
            response = []
            for chunk in self.llm.stream(prompt):
                if chunk == "<|eot_id|>":
                    break
                response.append(chunk)
            response = "".join(response)
        else:
            response = self.llm.invoke(prompt)
        return response

    def run(self, prompt, retry, max_retries=3, threshold=10):
        """Query the LLM, with optional retries for responses which are too long.

        Args:
            prompt(str): LLM prompt.
            retry(bool): Whether to retry queries after long LLM responses.
            max_retries(int): Maximum number of retries.
            threshold(int): Maximum number of words threshold to trigger a retry.

        Returns:
            str: LLM response
        """
        response = self.invoke(prompt)
        if retry and (len(response.split(" ")) > threshold):
            for _ in range(max_retries):
                new_prompt = self._retry_prompt.format(prompt=prompt, response=response)
                response = self.invoke(new_prompt)
                if len(response.split(" ")) <= threshold:
                    break
        return response.strip()

    @staticmethod
    def _remove_intro(topics):
        """Remove non-topic related introductory text from topic labels (e.g., labels that start with "Label:")

        Args:
            topics(list): LLM-generated topic labels.

        Returns:
            list: Cleaned topic labels.
        """
        cleaned = []
        for t in topics:
            if ":" in t:
                cleaned.append(t.split(":")[1].strip())
            else:
                cleaned.append(t)
        return cleaned

    def clean_llm_topics(self, topics):
        """Clean LLM-generated topic labels by removing text sequences which are not related to the label itself.

        Args:
            topics(list): LLM-generated topic labels.

        Returns:
            Cleaned topic labels.
        """
        topic_labels_cleaned = [t.split("[/INST]\n")[-1].strip() for t in topics]
        topic_labels_cleaned = self._remove_intro(topic_labels_cleaned)
        topic_labels_cleaned = [
            t.replace('"', "").strip() for t in topic_labels_cleaned
        ]
        topic_labels_cleaned = [t.split("\n")[0].strip() for t in topic_labels_cleaned]
        return topic_labels_cleaned
