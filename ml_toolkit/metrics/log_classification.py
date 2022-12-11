import numpy as np
import mlflow.pytorch
import matplotlib as mpl
import matplotlib.pyplot as plt
from ray.tune.integration.mlflow import mlflow_mixin
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, matthews_corrcoef,
                             ConfusionMatrixDisplay, balanced_accuracy_score)
from deepchem.metrics import bedroc_score

mpl.rcParams['figure.figsize'] = [10, 7]
mpl.use('Agg')


@mlflow_mixin
def log_classification(run: str, num: int, y_pred, y_true):
    """Calculate and log to mlflow classification metrics

    Args:
        run (str): Evaluation type
        num (int): Current epoch
        y_pred (_type_): Y pred
        y_true (_type_): Y true
    """
    y_pred = y_pred.astype(int)
    # * Calculate metrics
    acc = accuracy_score(y_pred, y_true)
    acc_bal = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_pred, y_true)
    prec = precision_score(y_pred, y_true)
    rec = recall_score(y_pred, y_true)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        roc = roc_auc_score(y_pred, y_true)
    except Exception:
        roc = 0

    bed_pred = np.zeros((y_pred.size, 2))
    bed_pred[np.arange(y_pred.size), y_pred] = 1
    bedroc = bedroc_score(y_true, bed_pred)

    # * Log on mlflow
    if mlflow.active_run() is not None:
        mlflow.log_metric(key=f"F1 Score-{run}", value=float(f1), step=num)
        mlflow.log_metric(key=f"Accuracy-{run}", value=float(acc), step=num)
        mlflow.log_metric(key=f"Accuracy balanced-{run}", value=float(acc_bal),
                          step=num)
        mlflow.log_metric(key=f"Precision-{run}", value=float(prec), step=num)
        mlflow.log_metric(key=f"Recall-{run}", value=float(rec), step=num)
        mlflow.log_metric(key=f"ROC-AUC-{run}", value=float(roc), step=num)
        mlflow.log_metric(key=f"MCC-{run}", value=float(mcc), step=num)
        mlflow.log_metric(key=f"Bedroc-{run}", value=float(bedroc), step=num)

    # * Plot confusion matrix for test run
    if (run == "test") or (run == "predict"):
        cm_raw = ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                                         colorbar=False,
                                                         cmap="Blues",
                                                         normalize=None)
        cm_norm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                                          colorbar=False,
                                                          cmap="Blues",
                                                          normalize="true")

        if mlflow.active_run() is not None:
            mlflow.log_figure(cm_raw.figure_, f"cm_{num}_raw.png")
            mlflow.log_figure(cm_norm.figure_, f"cm_{num}_norm.png")
            plt.close("all")

    if mlflow.active_run() is None:
        try:
            return [acc, acc_bal, f1, prec, rec, mcc, roc, bedroc,
                    cm_raw, cm_norm]
        except NameError:
            return [acc, acc_bal, f1, prec, rec, mcc, roc, bedroc, 0, 0]
