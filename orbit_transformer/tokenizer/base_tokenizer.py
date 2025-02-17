import pandas as pd


class BaseTokenizer:
    """
    A generic interface for a tokenizer.
    Subclasses should implement the transform() method
    that adds tokenized columns to a DataFrame.
    """
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
