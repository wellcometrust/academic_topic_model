import sys
import warnings

import awswrangler as wr
import click
import transformers
from tqdm import tqdm

try:
    import gcld3
except ImportError:
    pass

from pipeline.embedding.src.inference import EmbeddingModel

warnings.simplefilter(action="ignore", category=UserWarning)
transformers.logging.set_verbosity_error()


class AbstractEmbeddings(EmbeddingModel):
    """Child class for generating publication abstract sentence embeddings.

    By default uses BiomedNLP PubMedBERT model from HuggingFace.

    """

    def __init__(
        self,
        input_bucket,
        output_bucket,
        tokenizer_name,
        model_name,
        filter_language=False,
        parallelise=True
    ):
        """Initialise parent class and parameters.

        Args:
            input_bucket(str): S3 URI for abstracts text directory.
            output_bucket(str): S3 URI for embedding output directory.
            tokenizer_name(str): HuggingFace Hub tokenizer name.
            model_name(str): HuggingFace Hub model name.
            parallelise(bool): Parallelise over all available GPUs.

        """
        super().__init__(tokenizer_name, model_name, parallelise)

        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.filter_language = filter_language

    def language_filter(self, df):
        """Uses Google Compact Language Detector 3 (CLD3) to identify
        language of abstracts.

        Currently the only language supported by PubMedBERT is English.
        Non-English text is identified and filtered out.

        CLD3 is a shallow neural network developed for languge detection.

        Args:
            df(pd.DataFrame): Data frame containing abstract texts.

        Returns:
            pd.DataFrame: Pandas dataframe containing only English language
            abstracts.

        """
        # Language detection model from Google Chromium project.
        langauge_detector = gcld3.NNetLanguageIdentifier(
            min_num_bytes=0, max_num_bytes=1000
        )

        texts = df["abstract"]

        languages = []
        for text in texts:
            detected_language = langauge_detector.FindLanguage(text)
            languages.append(detected_language.language)

        df["language"] = languages
        df = df.loc[df["language"] == "en"]

        # Must reset to allow any merging with pd.Series later.
        df.reset_index(inplace=True)

        return df

    def read_parquet(self, year):
        """Reads parquet files containing publication abstracts from S3.

        Args:
            year(str): Publication year.

        Yields:
            pd.DataFrame: Pandas dataframe with publication id and abstracts.
            path: Full S3 URI path for parquet file.

        """
        bucket = f"{self.input_bucket}{year}/"
        parquet_paths = wr.s3.list_objects(bucket)
        for path in tqdm(parquet_paths):
            df = wr.s3.read_parquet(path, columns=["id", "abstract", "title"])

            df.dropna(subset=["abstract", "title"], how="all", inplace=True)
            df = df.loc[(df["abstract"] != "") & (df["title"] != "")].copy()

            df["length"] = (
                df["abstract"].str.split().str.len() + df["title"].str.split().str.len()
            )

            df["abstract"].fillna("", inplace=True)
            df["title"].fillna("", inplace=True)
            df["text"] = df["title"] + " " + df["abstract"]

            if self.filter_language:
                if 'gcld3' not in sys.modules:
                    raise ImportError(
                        'The gcld3 library is needed for language detection to work!'
                    )

                df = self.language_filter(df)

            # Remove very short texts.
            df = df.loc[df["length"] >= 20]

            # Must reset to allow adding pd.Series as a column later.
            df.reset_index(inplace=True, drop=True)

            yield df, path

    def embed_abstracts(self, start_year, end_year):
        """Generate sentence embeddings for publication abstracts.

        Embeddings are output to provided S3 path as parquet files.

        Args:
            start_year(int): Earliest publication year to include.
            end_year(int): Latest publication to include up to (exclusive).

        """
        print("\nGenerating embeddings...")
        years = [str(i) for i in range(int(start_year), int(end_year))]

        # Launch processes per target GPU.
        if self.gpu_parallelism:
            self.launch_processes()

        for year in years:
            print(year)

            for file, file_path in self.read_parquet(year):
                if file.empty:
                    continue

                embeddings = self.run_inference(file["abstract"].to_list())

                file["embeddings"] = embeddings

                file_name = file_path.split("/")[-1]
                output_path = f"{self.output_bucket}{year}/{file_name}"

                wr.s3.to_parquet(file, output_path, index=None)

        if self.gpu_parallelism:
            self.kill_processes()


@click.command()
@click.argument("start_year")
@click.argument("end_year")
@click.argument("input_s3_uri")
@click.argument("output_s3_uri")
@click.option("--parallelise", "-p", is_flag=True)
@click.option("--lang_filter", "-l", is_flag=True)
@click.option("--model", "-m", default="allenai/scibert_scivocab_uncased")
@click.option("--tokenizer", "-t", default="allenai/scibert_scivocab_uncased")
def infer_embeddings(
    start_year,
    end_year,
    input_s3_uri,
    output_s3_uri,
    parallelise=False,
    lang_filter=False,
    tokenizer="allenai/scibert_scivocab_uncased",
    model="allenai/scibert_scivocab_uncased",
):
    embedding_pipeline = AbstractEmbeddings(
        input_bucket=input_s3_uri,
        output_bucket=output_s3_uri,
        tokenizer_name=tokenizer,
        model_name=model,
        filter_language=lang_filter,
        parallelise=parallelise,
    )

    embedding_pipeline.embed_abstracts(start_year, end_year)


if __name__ == "__main__":
    infer_embeddings()
