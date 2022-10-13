# import os
# import uuid
# import argparse
# from ray import tune
# from pathlib import Path
# from ray.tune.schedulers import ASHAScheduler

# from gnn_toolkit.run import gnn_toolkit
# from gnn_toolkit.utils import get_config, TelegramReport
# from gnn_toolkit.evaluation import predict_model, explain_graph

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# LOGO = Path(str(os.getcwd())) / "assets" / "logo.txt"

# TODO WHOLE FILE

# # * Train class
# class TrainModel:
#     def __init__(self, cfg_file):
#         self.param_space = cfg_file
#         self.objective = gnn_toolkit
#         self.n_samples = cfg_file["RUN_RAY_SAMPLES"][0]
#         self.max_epoch = cfg_file["RUN_RAY_MAX_EPOCH"][0]
#         self.cpu = cfg_file["RUN_RAY_CPU"][0]
#         self.gpu = cfg_file["RUN_RAY_GPU"][0]
#         self.budget = cfg_file["RUN_RAY_TIME_BUDGET_S"][0]
#         self.resume = cfg_file["RUN_RAY_RESUME"][0]

#         self.mlflow_uri = cfg_file["RUN_MLFLOW_URI"][0]
#         self.telegram = cfg_file["RUN_TELEGRAM_VERBOSE"][0]
#         self.epoch = cfg_file["SOLVER_NUM_EPOCH"][0]
#         self.task = cfg_file["TASK"][0]

#         self.experiment_name = cfg_file["RUN_NAME"][0]
#         if self.experiment_name == "None":
#             self.experiment_name = str(uuid.uuid4()).split("-")[0]

#         cfg_file["MLFLOW_NAME"] = [self.experiment_name]
#         try:
#             mlflow.create_experiment(self.experiment_name)
#         except Exception:
#             pass

#         mlflow.set_tracking_uri(self.mlflow_uri)
#         mlflow.set_registry_uri("./mlruns")
#         cfg_file["mlflow"] = {"experiment_name": self.experiment_name,
#                               "tracking_uri": mlflow.get_tracking_uri()}

#     def execute(self):
#         print(f"Running on: {device}\n")
#         TelegramReport.start_eval(device, self.task,
#                                   self.n_samples, self.epoch,
#                                   self.telegram)

#         scheduler = ASHAScheduler(max_t=self.max_epoch,
#                                   grace_period=1, reduction_factor=1.2)

#         resources = {"cpu": self.cpu, "gpu": self.gpu}
#         result = tune.run(tune.with_parameters(self.objective),
#                           name=self.experiment_name,
#                           config=self.param_space,
#                           resources_per_trial=resources,
#                           num_samples=self.n_samples,
#                           scheduler=scheduler,
#                           metric="loss",
#                           mode="min",
#                           local_dir="./ray_results",
#                           trial_name_creator=trial_str_creator,
#                           trial_dirname_creator=trial_str_creator,
#                           verbose=3,
#                           resume=self.resume,
#                           raise_on_failed_trial=False)

#         best_trial = result.get_best_trial("loss", "min", "last")
#         print(f"\nBest parameters: {best_trial.config}\n")
#         print(f"Best loss: {best_trial.last_result['loss']}")
#         TelegramReport.end_eval(best_trial, self.telegram)

#         return best_trial


# def trial_str_creator(trial):
#     return "{}_{}".format(trial.trainable_name, trial.trial_id)


# def toolkit_parser():
#     parser = argparse.ArgumentParser(description="Train GNN models")
#     parser.add_argument("--cfg",
#                         type=str,
#                         required=True,
#                         help="Path to config file")
#     args = parser.parse_args()

#     return args


# def main():
#     # * Get parse
#     args = toolkit_parser()

#     # * Print logo
#     file = open(str(LOGO), mode="r")
#     logo = file.read()
#     file.close()
#     print(logo)

#     # * Load config and initialize experiment
#     cfg_file = get_config(args.cfg)
#     run_type = cfg_file["TYPE"][0].lower()

#     # * Run GNN-Toolkit
#     if run_type == "train":
#         ray_tune = TrainModel(cfg_file)
#         best_trial = ray_tune.execute()
#         _ = best_trial  # Remove flake8 F841 from best_trial

#     elif run_type == "predict":
#         model, loader = predict_model(cfg_file)

#         if cfg_file["EXPLAIN_USE"][0]:
#             explain_graph(model, loader, cfg_file)


# if __name__ == "__main__":
#     main()
