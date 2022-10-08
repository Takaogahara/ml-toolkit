import copy
import yaml
# Nonsense comment

train_param = {"TASK": ["Classification"],

               "RUN_NAME": ["None"],
               "RUN_MLFLOW_URI": ["http://localhost:5000"],
               "RUN_RAY_SAMPLES": [1],
               "RUN_RAY_MAX_EPOCH": [1],
               "RUN_RAY_GPU": [True],
               "RUN_RAY_TIME_BUDGET_S": [None],

               "DATA_PREMADE": [False],
               "DATA_PATH": ["/path/to/file.csv"],
               "DATA_SPLIT": [0.2],

               "MODEL_TYPE": ["sgdclassifier"],

               "OPTIMIZE_MLP_OPTIMIZER": ["SGD"],
               "OPTIMIZE_MLP_LEARNING_RATE": [0.001, 0.1],
               "OPTIMIZE_MLP_BATCH_SIZE": [64, 256]}


def get_config(yaml_path: str):
    """ Open YAML file and and process to parse in Ray Tune

    Args:
        yaml_path (str): path to YAML file

    Returns:
        Dict: Processed YAML file
    """
    with open(yaml_path, "r") as f:
        file = yaml.safe_load(f)

    content = {"TASK": file["TASK"]}
    parameters = copy.deepcopy(train_param)

    content_list = [file["RUN"], file["DATA"],
                    file["MODEL"], file["OPTIMIZE"]]

    for value in content_list:
        for item in value:
            key, value = list(item.items())[0]
            content[key] = value

    # * Auto complete config file
    default_params = ["TASK", "RUN", "DATA", "MODEL"]
    for key, value in list(content.items()):
        is_default = [df_param in key for df_param in default_params]
        is_default = any(is_default)

        if (not is_default) and (key != "OPTIMIZE_MLP_OPTIMIZER"):
            parameters[key] = tuple(value)
        else:
            parameters[key] = value

    return parameters


def extract_configs(config_file: dict):
    """Extract YAML file

    Args:
        config_file (dict): Processed YAML file

    Returns:
        Dict: Extracted YAML file
    """

    # * Extract config file
    parameters_new = {}
    for key, value in config_file.items():
        if isinstance(value, list):
            parameters_new[key] = value[0]
        else:
            parameters_new[key] = value

    return parameters_new
