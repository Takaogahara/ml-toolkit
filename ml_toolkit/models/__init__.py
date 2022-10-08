import torch
from skorch import NeuralNetClassifier

from ml_toolkit.models.model_selection import ClassMLP, get_sklearn_models


def get_model(cfg_file: dict):
    task = cfg_file["TASK"].lower()
    model_type = cfg_file["MODEL_TYPE"].lower()

    available_tasks = ["classification"]
    if task not in available_tasks:
        raise RuntimeError(f"Task not supported. Available: {available_tasks}")

    if model_type == "mlp":
        model = _select_torch_model(cfg_file)

    else:
        model = _select_sk_models(cfg_file)

    return model


def _select_sk_models(cfg_file: dict):
    model = cfg_file["MODEL_TYPE"].lower()
    models_dict = get_sklearn_models()

    keys = list(models_dict.keys())
    if model not in keys:
        raise RuntimeError(f"Function not supported. Available: {keys}")

    function = models_dict[model]

    return function


def _select_torch_model(cfg_file: dict):
    model = cfg_file["MODEL_TYPE"].lower()
    if model not in ["mlp"]:
        raise RuntimeError("Function not supported. Available: 'MLP'")

    ray_gpu = cfg_file["RUN_RAY_GPU"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not ray_gpu:
        device = torch.device("cpu")

    max_epoch = cfg_file["RUN_RAY_MAX_EPOCH"]
    optimizer = cfg_file["OPTIMIZE_MLP_OPTIMIZER"].lower()
    if optimizer == "sgd":
        optim = torch.optim.SGD
    elif optimizer == "adam":
        optim = torch.optim.Adam
    else:
        keys = ["SGD", "Adam"]
        raise RuntimeError(f"Optimizer not supported. Available: {keys}")

    function = NeuralNetClassifier(ClassMLP, device=device, verbose=1,
                                   max_epochs=max_epoch,
                                   train_split=None,
                                   iterator_train__shuffle=True,
                                   batch_size=64,
                                   optimizer=optim,
                                   lr=0.1)

    return function
