import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from ml_toolkit.dataloader.load_csv import load_csv_data
SEED = 8


def data_selection(cfg_file: dict):
    """Main function to load data

    Args:
        cfg_file (dict): dict like config parameters

    Raise:
        RuntimeError: If task not avaliable

    Returns:
        train (tuple): train data in tuple format (X, y)
        test (tuple): Test data in tuple format (X, y)
    """
    task = cfg_file["TASK"].lower()
    model_type = cfg_file["MODEL_TYPE"].lower()

    available_tasks = ["classification", "regression"]
    if task not in available_tasks:
        raise RuntimeError(f"Task not supported. Available: {available_tasks}")

    # TODO Verificar utilidade
    if model_type == "mlp":
        train, test = load_csv_data(cfg_file)

    else:
        train, test = _load_csv_sk(cfg_file)

    return train, test


def _load_csv_sk(cfg_file):
    """Function to load csv datasets

    Args:
        cfg_file (dict): dict like config parameters

    Returns:
        train (tuple): train data in tuple format (X, y)
        test (tuple): Test data in tuple format (X, y)
    """
    split = cfg_file["DATA_SPLIT"]
    premade = cfg_file["DATA_PREMADE"]
    path = cfg_file["DATA_PATH"]
    path = str(Path(path))

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
        y_train = y_train.astype(np.int64)
        x_train = df_train.drop(columns=["Label"]).to_numpy()
        x_train = x_train.astype(np.float32)

        # Get test subset
        df_test = dataframe[dataframe["Set"].str.lower() == "test"]
        df_test = df_test.drop(columns=["Set"])
        if len(df_test) == 0:
            warnings.warn("Test subset has no len")

        y_test = df_test[["Label"]].to_numpy().flatten()
        y_train = y_train.astype(np.int64)
        x_test = df_test.drop(columns=["Label"]).to_numpy()
        x_train = x_train.astype(np.float32)

        # Return
        train = (x_train, y_train)
        test = (x_test, y_test)
        return train, test

    else:
        # Read CSV file
        dataframe = pd.read_csv(path)
        dataframe = dataframe.drop(columns=["ID", "Set"])

        y = dataframe[["Labels"]].to_numpy().flatten()
        y = y.astype(np.int64)
        X = dataframe.drop(columns=["Labels"]).to_numpy()
        X = X.astype(np.float32)

        x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=split,
                                                            random_state=SEED,
                                                            stratify=y)

        # Return
        train = (x_train, y_train)
        test = (x_test, y_test)
        return train, test
