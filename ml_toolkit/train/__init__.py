from ml_toolkit.train.fit_sk import train_sk_models
# from ml_toolkit.train.fit_torch import train_torch_models
from ml_toolkit.train.fit_mlp import train_torch_models


def fit(train: tuple, test: tuple, model, cfg_file: dict):
    """Main function to fit estimators

    Args:
        train (tuple): Loaded train data
        model: Selected estimator
        cfg_file (dict): dict like config parameters

    Raise:
        RuntimeError: If task not avaliable

    Returns:
        best_params (dict): Best parameters
    """
    if "BasicMLP" in str(model.__class__):
        best_loss = train_torch_models(train, test, model, cfg_file)

    else:
        best_loss = train_sk_models(train, test, model, cfg_file)

    return best_loss
