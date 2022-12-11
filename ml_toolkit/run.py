import ray
import mlflow
from ray.tune.integration.mlflow import mlflow_mixin

from ml_toolkit.utils import extract_configs, DEFAULT_PARAMS
from ml_toolkit.dataloader import data_selection
from ml_toolkit.models import model_selection
from ml_toolkit.train import fit


@mlflow_mixin
def _log_parameters(parameters):
    model_type = parameters["MODEL_TYPE"].lower()

    mlp_val = ["OPTIMIZE_MLP_OPTIMIZER", "OPTIMIZE_MLP_LEARNING_RATE",
               "OPTIMIZE_MLP_BATCH_SIZE", "OPTIMIZE_EMBEDDING_SIZE",
               "OPTIMIZE_DROPOUT_RATE", "OPTIMIZE_NUM_LAYERS"]

    # * Extract optimizer variables
    optim_dict = {}
    for key, value in list(parameters.items()):
        is_default = [df_param in key for df_param in DEFAULT_PARAMS]
        is_default = any(is_default)

        if (not is_default):
            optim_dict[key] = value

    del optim_dict["OPTIMIZE_FEAT_SIZE"]

    if model_type == "mlp":
        for key in optim_dict.keys():
            if key in mlp_val:
                mlflow.log_param(key, optim_dict[key])

    else:
        for key in optim_dict.keys():
            if key not in mlp_val:
                mlflow.log_param(key, optim_dict[key])


@mlflow_mixin
def ml_toolkit(config, checkpoint_dir=None):
    """Start run

    Args:
        parameters (dict): Ray Tune parsed parameters

    Returns:
        int: Best loss
    """
    if ray.tune.is_session_enabled():
        mlflow.set_tag("mlflow.runName", ray.tune.get_trial_name())
        with ray.tune.checkpoint_dir(step=1) as checkpoint_dir:
            mlflow.set_tag("Ray Tune", checkpoint_dir)
    parameters = extract_configs(config)

    # * Load DataLoaders
    loader_train, loader_test = data_selection(parameters)

    # * Get information
    if isinstance(loader_train, tuple):
        parameters["OPTIMIZE_FEAT_SIZE"] = loader_train[0].shape[1]
    else:
        parameters["OPTIMIZE_FEAT_SIZE"] = len(loader_train.dataset[0][0])

    # * Load Model
    model = model_selection(parameters)

    # * Log parameters in mlflow
    _log_parameters(parameters)

    # * Train
    best_loss = fit(loader_train, loader_test, model, parameters)

    return best_loss


# # TODO REMOVE BEFORE USING MAIN.PY
# parameters = {"TASK": ["Classification"],
#               "TYPE": ["Train"],

#               "RUN_NAME": ["None"],
#               "RUN_MLFLOW_URI": ["http://localhost:5000"],
#               "RUN_RAY_SAMPLES": [1],
#               "RUN_RAY_MAX_EPOCH": [5],
#               "RUN_RAY_CPU": [2],
#               "RUN_RAY_GPU": [1],
#               "RUN_RAY_TIME_BUDGET_S": ["None"],
#               "RUN_RAY_RESUME": [False],

#               "DATA_PREMADE": [False],
#               "DATA_PATH": [("/media/takaogahara/external/1_datasets/"
#                              "MMlab/verissimo_luna/dummy_luna.csv")],
#               "DATA_SPLIT": [0.2],

#             #   "MODEL_TYPE": ["MLP"],
#               "MODEL_TYPE": ["sgdclassifier"],

#               "OPTIMIZE_MLP_OPTIMIZER": ["SGD"],
#               "OPTIMIZE_MLP_LEARNING_RATE": [0.1],
#               "OPTIMIZE_MLP_BATCH_SIZE": [64],
#               "OPTIMIZE_EMBEDDING_SIZE": [16],
#               "OPTIMIZE_DROPOUT_RATE": [0.2],
#               "OPTIMIZE_NUM_LAYERS": [2],
#               "alpha": [0.1],
#               "epsilon": [0.01]}

# x = ml_toolkit(parameters)
