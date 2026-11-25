"""Loading the corpus and splitting it without leakage (Lab 3A)."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config
from .preprocessing import preprocess


def load_projects(path=config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].map(preprocess)          # one preprocessing truth
    df["label"] = df["category"].map(config.CATEGORIES.index)
    df["length"] = df["clean_text"].str.split().str.len().map(lambda n: "short" if n <= 8 else "long")
    return df


def grouped_split(df: pd.DataFrame, test_size=config.TEST_SIZE, seed=config.SEED):
    """Split by team_id so two descriptions of one project never straddle train/test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["team_id"]))
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    shared = set(train_df["team_id"]) & set(test_df["team_id"])
    assert not shared, f"Leakage: teams {shared} are in both train and test"
    return train_df, test_df
