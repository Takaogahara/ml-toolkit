# import ray
# import mlflow.sklearn as mlflow_sk
from sklearn.metrics import accuracy_score
from ray.tune.integration.mlflow import mlflow_mixin

from ml_toolkit.train import fit
from ml_toolkit.data import load_data
from ml_toolkit.models import get_model
from ml_toolkit.utils import extract_configs

# TODO REMOVE BEFORE USING MAIN.PY
from ml_toolkit.utils import get_config


@mlflow_mixin
def ml_toolkit(config):
    """ Run ML-Toolkit

    Args:
        parameters (dict): Dict like parameters

    Returns:
        dict: Best parameters
    """
    # TODO Integrate MLFlow
    # TODO Fix default None run name
    # if ray.tune.is_session_enabled():
    #     mlflow_sk.set_tag("mlflow.runName", ray.tune.get_trial_name())
    #     with ray.tune.checkpoint_dir(step=1) as checkpoint_dir:
    #         mlflow_sk.set_tag("Ray Tune", checkpoint_dir)
    parameters = extract_configs(config)

    # * Load DataLoaders
    loader_train, loader_test = load_data(parameters)
    parameters["OPTIMIZE_FEAT_SIZE"] = loader_train[0].shape[1]

    # * Load Model
    model = get_model(parameters)

    # * Log Parameters in mlflow
    # for key in parameters.keys():
    #     mlflow_sk.log_param(key, parameters[key])

    # * Train
    best_params = fit(loader_train, model, parameters)

    # * Validate
    # TODO Remove or create function
    x_train, y_train = loader_train
    x_test, y_test = loader_test

    clf = model(**best_params)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    return best_params


# TODO REMOVE BEFORE USING MAIN.PY
path = "/media/takaogahara/data/projects/ml-toolkit/configs/ray_train.yaml"
config = get_config(path)

accuracy = ml_toolkit(config)
