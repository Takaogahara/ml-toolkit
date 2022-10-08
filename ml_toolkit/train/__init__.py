import pickle
from ray.tune.sklearn import TuneSearchCV


def fit(train: list, model, cfg_file: dict):
    task = cfg_file["TASK"].lower()

    available_tasks = ["classification"]
    if task not in available_tasks:
        raise RuntimeError(f"Task not supported. Available: {available_tasks}")

    if "skorch" in str(model.__class__):
        best_params = _train_torch_models(train, model, cfg_file)

    else:
        best_params = _train_sk_models(train, model, cfg_file)

    return best_params


def _train_sk_models(train: list, model, cfg_file: dict):
    x_train, y_train = train
    run_name = cfg_file["RUN_NAME"]
    trials = cfg_file["RUN_RAY_SAMPLES"]
    max_epoch = cfg_file["RUN_RAY_MAX_EPOCH"]
    ray_gpu = cfg_file["RUN_RAY_GPU"]
    budget = cfg_file["RUN_RAY_TIME_BUDGET_S"]
    if budget.lower() == "none":
        budget = None

    optim_grid = {}
    for key in list(cfg_file.keys()):
        lower = "".join([chr for chr in key if key.islower()])
        if len(lower) > 0:
            optim_grid[lower] = cfg_file[lower]

    tune_search = TuneSearchCV(model(), optim_grid,
                               search_optimization="bayesian",
                               early_stopping=True, random_state=8,
                               n_trials=trials, max_iters=max_epoch,
                               n_jobs=1, cv=None, verbose=1,
                               local_dir="./ray_results",
                               name=run_name, use_gpu=ray_gpu,
                               time_budget_s=budget, loggers=[])

    tune_search.fit(x_train, y_train)
    params = tune_search.best_params_

    # TODO
    filename = "./estimator.pkl"
    pickle.dump(tune_search.best_estimator_, open(filename, 'wb'))

    return params


def _train_torch_models(train: list, model, cfg_file: dict):
    x_train, y_train = train
    run_name = cfg_file["RUN_NAME"]
    trials = cfg_file["RUN_RAY_SAMPLES"]
    ray_gpu = cfg_file["RUN_RAY_GPU"]
    budget = cfg_file["RUN_RAY_TIME_BUDGET_S"]
    if budget.lower() == "none":
        budget = None

    max_epoch = cfg_file["RUN_RAY_MAX_EPOCH"]
    batch_size = cfg_file["OPTIMIZE_MLP_BATCH_SIZE"]
    lr = cfg_file["OPTIMIZE_MLP_LEARNING_RATE"]

    feature_size = cfg_file["OPTIMIZE_FEAT_SIZE"]
    embedding_units = cfg_file["OPTIMIZE_EMBEDDING_SIZE"]
    dropout = cfg_file["OPTIMIZE_DROPOUT_RATE"]
    n_layers = cfg_file["OPTIMIZE_NUM_LAYERS"]

    optim_grid = {"lr": lr, "batch_size": batch_size,
                  "module__features": [feature_size],
                  "module__embedding": embedding_units,
                  "module__dropout": dropout,
                  "module__n_layers": n_layers}

    tune_search = TuneSearchCV(model, optim_grid,
                               search_optimization="bayesian",
                               early_stopping=True, random_state=8,
                               n_trials=trials, max_iters=max_epoch,
                               scoring="accuracy", cv=None,
                               n_jobs=1, verbose=1,
                               local_dir="./ray_results",
                               name=run_name, use_gpu=ray_gpu,
                               time_budget_s=budget, loggers=[])

    tune_search.fit(x_train, y_train)
    params = tune_search.best_params_

    # TODO
    filename = "./estimator.pkl"
    pickle.dump(tune_search.best_estimator_, open(filename, 'wb'))

    return params
