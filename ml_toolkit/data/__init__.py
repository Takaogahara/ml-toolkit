import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
SEED = 8


def load_data(cfg_file: dict):
    task = cfg_file["TASK"].lower()
    model_type = cfg_file["MODEL_TYPE"].lower()

    available_tasks = ["classification"]
    if task not in available_tasks:
        raise RuntimeError(f"Task not supported. Available: {available_tasks}")

    if model_type == "mlp":
        train, test = _load_torch_data(cfg_file)

    else:
        train, test = _load_sklearn_data(cfg_file)

    return train, test


def _load_torch_data(cfg_file):
    _ = cfg_file["DATA_SPLIT"]
    _ = cfg_file["DATA_PREMADE"]
    _ = cfg_file["DATA_PATH"]  # TODO USE POSIX

    X, y = make_classification(1000, 20, n_informative=10, random_state=SEED)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    return (X, y), ()


def _load_sklearn_data(cfg_file: dict):
    split = cfg_file["DATA_SPLIT"]
    premade = cfg_file["DATA_PREMADE"]
    path = cfg_file["DATA_PATH"]  # TODO USE POSIX

    if premade:
        # Read CSV file
        dataframe = pd.read_csv(path)
        dataframe = dataframe.drop(columns=["ID"])

        # Get train subset
        df_train = dataframe[dataframe["Set"].str.lower() == "train"]
        df_train = df_train.drop(columns=["Set"])
        if len(df_train) == 0:
            warnings.warn("Train subset has no len")

        y_train = df_train[["Label"]].to_numpy().flatten()
        x_train = df_train.drop(columns=["Label"]).to_numpy()

        # Get test subset
        df_test = dataframe[dataframe["Set"].str.lower() == "test"]
        df_test = df_test.drop(columns=["Set"])
        if len(df_test) == 0:
            warnings.warn("Test subset has no len")

        y_test = df_test[["Label"]].to_numpy().flatten()
        x_test = df_test.drop(columns=["Label"]).to_numpy()

        # Return
        train = (x_train, y_train)
        test = (x_test, y_test)
        return train, test

    else:
        # Read CSV file
        dataframe = pd.read_csv(path)
        dataframe = dataframe.drop(columns=["ID", "Set"])

        y = dataframe[["Label"]].to_numpy().flatten()
        X = dataframe.drop(columns=["Label"]).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=split,
                                                        random_state=SEED,
                                                        stratify=y)

    train = (x_train, y_train)
    test = (x_test, y_test)
    return train, test
