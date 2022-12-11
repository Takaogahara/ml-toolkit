import mlflow.pytorch
from ray.tune.integration.mlflow import mlflow_mixin

from ml_toolkit.metrics.log_classification import log_classification


@mlflow_mixin
def log_metrics(task: str, eval_type: str,
                current_epoch: int, y_pred, y_true):
    """External function for compute metrics

    Args:
        task (str): Current task
        eval_type (str): Evaluation type
        current_epoch (int): Current epoch
        y_pred (_type_): Y pred
        y_true (_type_): Y true
    """
    if task.lower() == "classification":
        metric = log_classification(eval_type, current_epoch,  y_pred, y_true)
        if mlflow.active_run() is None:
            return metric

    else:
        raise NotImplementedError
