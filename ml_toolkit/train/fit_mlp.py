import os
import ray
import torch
import numpy as np
from ray import tune
from tqdm import tqdm
import mlflow.pytorch
from ray.tune.integration.mlflow import mlflow_mixin

from ml_toolkit import get_root
from ml_toolkit.metrics import log_metrics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = get_root()


@mlflow_mixin
def train_torch_models(loader_train, loader_test, model, cfg_file):
    """Control training phase

    Args:
        model (_type_): Generated model
        cfg_file (dict): cfg_file YAML file
        optimizer (_type_): Generated optimizer
        loss_fn (_type_): Generated loss function
        loader_train (_type_): Train dataloaders
        loader_test (_type_): Test dataloaders
        scheduler (_type_): Pytorch scheduler

    Returns:
        int: Best loss
    """
    num_epoch = cfg_file["RUN_RAY_MAX_EPOCH"]
    optimizer = _get_optimizer(model, cfg_file)
    loss_fn = _get_lossfn(cfg_file)

    # torch_scheduler = 0

    # * Start run
    # print("\n############################################## Start")
    best_loss = 1000

    for epoch in range(1, num_epoch+1):
        # * TRAIN
        model.train()
        loss_tr = _train_epoch(cfg_file, model, optimizer, loss_fn,
                               loader_train, epoch, num_epoch)
        mlflow.log_metric(key="Train loss", value=float(loss_tr), step=epoch)

        # * TEST
        model.eval()
        loss_ts = _test_epoch(cfg_file, model, loss_fn,
                              loader_test, epoch)
        mlflow.log_metric(key="Test loss", value=float(loss_ts), step=epoch)

        # torch_scheduler.step()

        # * Save best model
        if loss_ts < best_loss:
            best_loss = loss_ts

            # * Save checkpoint
            if ray.tune.is_session_enabled():
                with tune.checkpoint_dir(step=1) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(),
                                optimizer.state_dict()), path)

            # * Save Model
            requirements = os.path.join(ROOT, "requirements.txt")
            mlflow.pytorch.log_model(model, "model",
                                     pip_requirements=requirements)

        if ray.tune.is_session_enabled():
            tune.report(loss=loss_ts)
    # print(f"Finishing training with best test loss: {best_loss}")
    # print("############################################## End\n")

    return loss_ts


def _train_epoch(cfg_file, model, optimizer, loss_fn,
                 loader, current_epoch: int, num_epoch: int):
    """Train model for one epoch

    Args:
        cfg_file (dict): cfg_file YAML file
        model (_type_): Generated model
        optimizer (_type_): Generated optimizer
        loss_fn (_type_): Generated loss function
        loader (_type_): Dataloader
        current_epoch (int): Current epoch
        num_epoch (int): Total epochs

    Returns:
        float: Loss
    """
    task = cfg_file["TASK"]

    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 1

    txt = f"Epoch {current_epoch}/{num_epoch}"
    unit = "batch"
    with tqdm(loader, ncols=120, unit=unit,
              desc=txt, disable=True) as bar:
        for batch in bar:

            # * Use GPU
            batch[0].to(device)

            # * Reset gradients
            optimizer.zero_grad()

            # * Passing the node features and the connection info
            pred = model(X=batch[0].float())

            # * Calculating the loss and gradients
            loss = loss_fn(torch.squeeze(pred),
                           torch.squeeze(batch[1].float()))
            loss.backward()

            # * Clip gradients to aviod exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # * Update gradients
            optimizer.step()

            # * Update tracking
            running_loss += loss.detach().item()
            step += 1

            # * Save pred results
            all_preds.append(
                np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
            all_labels.append(batch[1].cpu().detach().numpy())

            # * Update progress bar
            partial = running_loss/step
            bar.set_postfix_str(f"loss: {round(partial, 5)}")

        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()

        log_metrics(task, "train", current_epoch, all_preds, all_labels)

        return running_loss/step


@torch.no_grad()
def _test_epoch(parameters, model, loss_fn, loader,
                current_epoch: int):
    """Test model for one epoch

    Args:
        parameters (dict): Parameters YAML file
        model (_type_): Generated model
        loss_fn (_type_): Generated loss function
        loader (_type_): Dataloader
        current_epoch (int): Current epoch

    Returns:
        float: Loss
    """
    task = parameters["TASK"]

    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0

    for batch in loader:
        # * Use GPU
        batch[0].to(device)

        # * Passing the node features and the connection info
        pred = model(X=batch[0].float())

        # * Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred),
                       torch.squeeze(batch[1].float()))

        # * Update tracking
        running_loss += loss.item()
        step += 1

        # * Save pred results
        all_preds.append(
            np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch[1].cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    log_metrics(task, "test", current_epoch, all_preds, all_labels)

    return running_loss/step


def _get_optimizer(model, parameters: dict):
    """Create optimizer

    Args:
        model (_type_): Generated model
        parameters (dict): Parameters YAML file

    Raises:
        RuntimeError: If selection don't match

    Returns:
        _type_: Optimizer
    """
    optim_name = parameters["OPTIMIZE_MLP_OPTIMIZER"]
    lr = parameters["OPTIMIZE_MLP_LEARNING_RATE"]

    available_optim = ["sgd", "adam"]

    if optim_name.lower() not in available_optim:
        raise RuntimeError("Wrong optimizer, Available: \n"
                           f"{available_optim}")

    if optim_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr)

    elif optim_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr)

    return optimizer


def _get_lossfn(parameters: dict):
    """Create loss function

    Args:
        parameters (dict): Parameters YAML file

    Returns:
        _type_: Loss function
    """

    task = parameters["TASK"]
    optim_weight_input = 1
    # TODO: FIX optim_weight_input

    if task == "Classification":
        #  * < 1 increases precision, > 1 increases recall
        weight = torch.tensor([optim_weight_input],
                              dtype=torch.float32).to(device)

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    elif task == "Regression":
        loss_fn = torch.nn.MSELoss()

    else:
        raise RuntimeError("Wrong task type, Available: \n"
                           "Classification, Regression")

    return loss_fn
