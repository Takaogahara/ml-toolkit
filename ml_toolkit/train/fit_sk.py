import ray
from ray import tune
from sklearn import metrics
from ray.tune.integration.mlflow import mlflow_mixin

from ml_toolkit.metrics import log_metrics


@mlflow_mixin
def train_sk_models(train: tuple, test: tuple, model, cfg_file: dict):
    """Function to fit sklearn like estimators

    Args:
        train (tuple): Loaded train data
        model: Selected estimator
        cfg_file (dict): dict like config parameters

    Returns:
        params (dict): Best parameters
    """
    task = cfg_file["TASK"].lower()
    available_tasks = ["classification"]
    if task not in available_tasks:
        raise RuntimeError(f"Task not supported. Available: {available_tasks}")

    x_train, y_train = train
    x_test, y_test = test

    optim_grid = {}
    for key in list(cfg_file.keys()):
        lower = "".join([chr for chr in key if key.islower()])
        if (len(lower) > 0) and (lower != "mlflow"):
            optim_grid[lower] = cfg_file[lower]

    estimator = model(**optim_grid)
    estimator.fit(x_train, y_train)

    # * Log metrics
    y_pred_train = estimator.predict(x_train)
    log_metrics("classification", "train", 0, y_pred_train, y_train)

    y_pred_test = estimator.predict(x_test)
    log_metrics("classification", "test", 0, y_pred_test, y_test)

    # * loss
    loss = metrics.matthews_corrcoef(y_test, y_pred_test)
    loss = (-1 * loss)

    if ray.tune.is_session_enabled():
        tune.report(loss=loss)

    return loss
