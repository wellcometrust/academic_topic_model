import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import awswrangler as wr
from tqdm import tqdm


class BaseModel:
    """Methods to load and save embeddings data from parquet."""

    def __init__(
        self,
        input_bucket,
        embedding_col,
        id_col="publication_id",
        schema=None,
        additional_cols=None,
    ):
        """Initialise base model and parameters.

        Args:
            input_bucket(list, str): S3 URI or list of S3 URIs for text embedding directory.
            embedding_col(str): Name of column containing embeddings.
            id_col(str): Name of column containing publication IDs.
            schema(pa.schema): Pyarrow schema for embeddings data.
            additional_cols(Optional[list]): Names of any additional columns to load.

        """
        if isinstance(input_bucket, str):
            input_bucket = [input_bucket]
        self._input_bucket = input_bucket
        self.fpaths = [
            fpath for ib in self._input_bucket for fpath in wr.s3.list_objects(ib)
        ]
        self.id_col = id_col

        self.data = self.load_embeddings(embedding_col, additional_cols, schema=schema)
        self._embedding_col = embedding_col

    def load_embeddings(self, embedding_col, additional_cols=None, schema=None):
        """Load embeddings from parquet files on s3.

        Args:
            embedding_col(str): Name of column containing embeddings.
            additional_cols(Optional[list]): Names of any additional columns to load.
            schema(pa.schema): Pyarrow schema for embeddings data.

        Returns:
            pd.DataFrame: Publication IDs and embeddings (and additional data, if requested).
        """

        columns = [self.id_col, embedding_col]
        if additional_cols:
            columns.extend(additional_cols)

        print("\nLoading embeddings from parquet...")

        df = wr.s3.read_parquet(self.fpaths, columns=columns, schema=schema)
        df.rename(columns={self.id_col: "id"}, inplace=True)
        df.set_index("id", inplace=True)
        df.dropna(inplace=True)
        return df

    def _save(self, fpath, new_colname, embeddings_dict, na_value):
        """Append column to embeddings dataframe from file.

        Args:
            fpath(str): Path to embeddings parquet file.
            new_colname(str): Name of column to append.
            embeddings_dict(dict): Dictionary containing data to append.
            na_value(Optional[Any]): Value to fill NAs with (optional).

        """
        df = wr.s3.read_parquet(fpath)
        df[new_colname] = list(map(embeddings_dict.get, df[self.id_col].tolist()))
        if na_value is not None:
            df[new_colname].fillna(na_value)
        try:
            wr.s3.to_parquet(df=df, path=fpath)
        except:
            print(fpath)
            wr.s3.to_parquet(
                df=df,
                path=f"{Path(fpath).parent / Path(fpath).stem}_{new_colname}.parquet",
            )

    def save(self, colname, na_value=None, **kwargs):
        """Append data to embeddings parquet files.

        Args:
            colname(str): Name of column containing data to save.
            **kwargs: Keyword arguments to attach to the column name.
        """
        embeddings_dict = self.data[colname].to_dict()
        new_colname = colname
        if kwargs:
            for k, v in kwargs.items():
                new_colname += f"_{v}{k}"

        with ThreadPoolExecutor(os.cpu_count()) as pool:
            list(
                tqdm(
                    pool.map(
                        partial(
                            self._save,
                            new_colname=new_colname,
                            embeddings_dict=embeddings_dict,
                            na_value=na_value,
                        ),
                        self.fpaths,
                    ),
                    total=len(self.fpaths),
                )
            )
