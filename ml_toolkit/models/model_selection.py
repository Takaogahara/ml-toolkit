from ml_toolkit.models.models import SK_MODELS, BasicMLP


def model_selection(cfg_file: dict):
    """Main function to create estimators

    Args:
        cfg_file (dict): dict like config parameters

    Raise:
        RuntimeError: If task not avaliable

    Returns:
        model: Loaded model
    """
    model_type = cfg_file["MODEL_TYPE"].lower()

    if model_type == "mlp":
        model = _select_torch_model(cfg_file)

    else:
        model = _select_sk_models(cfg_file)

    return model


def _select_sk_models(cfg_file: dict):
    """Function to load sklearn like estimators

    Args:
        cfg_file (dict): dict like config parameters

    Returns:
        model_fn: Sklearn like estimator
    """

    task = cfg_file["TASK"].lower()
    available_tasks = ["classification", "regression"]
    if task not in available_tasks:
        raise RuntimeError(f"Task not supported. Available: {available_tasks}")

    model = cfg_file["MODEL_TYPE"].lower()

    keys = list(SK_MODELS.keys())
    if model not in keys:
        raise RuntimeError(f"Function not supported. Available: {keys}")

    model_fn = SK_MODELS[model]

    return model_fn


def _select_torch_model(cfg_file: dict):
    """Function to load Skorch like estimators (MLP)

    Args:
        cfg_file (dict): dict like config parameters

    Returns:
        function: Skorch like estimator
    """
    task = cfg_file["TASK"].lower()
    available_tasks = ["classification", "regression"]
    if task not in available_tasks:
        raise RuntimeError(f"Task not supported. Available: {available_tasks}")

    feature_size = cfg_file["OPTIMIZE_FEAT_SIZE"]
    embedding_units = cfg_file["OPTIMIZE_EMBEDDING_SIZE"]
    n_layers = cfg_file["OPTIMIZE_NUM_LAYERS"]
    dropout = cfg_file["OPTIMIZE_DROPOUT_RATE"]

    model_fn = BasicMLP(feature_size, embedding_units, n_layers, dropout)

    return model_fn
