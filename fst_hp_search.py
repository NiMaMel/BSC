import os
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner,ThresholdPruner

import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.fst_model import Model
from data.dataset_modul import Dataset
from utils.preprocessing import preprocessing
from utils.utils import  data_split, hpSearchTrain, auc_score, dauc_pr

seed = 42

np.random.seed(seed)
torch.manual_seed(seed)

# cuda setting
if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    device = torch.device('cuda:%d' % device_id)
else:
    device = torch.device('cpu')
print(f"device set:{torch.cuda.get_device_name(device_id)}\n")

# 1. Load Dataset and create Tiplet-Df
sider = pd.read_csv("data/datasets/sider.csv")
triplets = [(i,j,row.iloc[j]) for i,row in sider.iterrows() for j in range(1,len(row))]
triplet_df = pd.DataFrame(triplets, columns=['mol_id', 'target_id', 'label'])

# 2. Preprocessing
data = preprocessing(sider)

def objective(trial):

    heads = [1, 2, 4,8]
    out_dims =  [dim for dim in range(12, 1125) if all((dim * 2) % head == 0 for head in heads)]

    params = {
        "opt_lr": trial.suggest_float("opt_lr", 1e-7, 1e-5,log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3,log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "input_dim": 2248,
        "hidden_dim": trial.suggest_int("hidden_dim", 1124, 4068, step=128),
        "output_dim": trial.suggest_categorical("output_dim", out_dims),
        "t_hidden_dim": trial.suggest_categorical('t_hidden_dim', [256, 512, 1024, 2048]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "t_num_layers": trial.suggest_int('t_num_layers', 1, 4),
        "num_heads": trial.suggest_categorical('num_heads', heads),
        "p": trial.suggest_float("p", 0.1, 0.5)
        }

    default_params = {
        "opt_lr": 1e-05,
        "weight_decay": 1e-05,
         "batch_size": 32,
         "input_dim": 2248,
         "hidden_dim": 1124,
         "output_dim": 1124,
         "t_hidden_dim": 2048,
         "num_layers": 2,
         "t_num_layers": 1,
         "num_heads": 4,
         "p": 0.1,
         "encodeLabels": False}

    # 3. Initialize Model
    model_params = {k: v for k, v in params.items() if k not in ['opt_lr', 'weight_decay']} #trial params
    #model_params = {k: v for k, v in default_params.items() if k not in ['opt_lr', 'weight_decay']} #default params
    model = Model(**model_params)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=params["opt_lr"], weight_decay=params["weight_decay"])

    # 4. Train-split and Dataloader

    train_triplet, val_triplet, test_triplet = data_split(triplet_df, seed)

    train_set = Dataset(data, sider, train_triplet, supp=8, seed=seed)
    val_set = Dataset(data, sider, val_triplet, supp=8, train=False, seed=seed)
    test_set = Dataset(data, sider, test_triplet, supp=8, train=False, seed=seed) # not used

    BATCH_SIZE = params["batch_size"]
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True) # not used

    # 5. Modeltraining

    train_config = {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "max_epochs": 50,
        "patience": 4,
        "log_interval": 100,
        "val_interval": 2,  # set higher to fasten up the process of training
        "save_path": f"{os.path.join('hp_search_runs/models', f'model_hps')}.mdl"
    }

    writer = SummaryWriter()
    hpSearchTrain(train_config, writer, train_loader, val_loader, device, BATCH_SIZE, trial = trial)

    # 6. Evaluation

    # load best model
    model.load_state_dict(torch.load(train_config["save_path"]))

    auc = auc_score(model, val_loader, device)
    daucPr = dauc_pr(model, val_loader, device)

    return daucPr #auc

if __name__ == "__main__":

    print("Trial started...\n")

    #pruner = MedianPruner()
    pruner= ThresholdPruner(lower=0.22, n_warmup_steps=15)  # Prune if metric < lower after n_warmup_steps epcohs

    study = optuna.create_study(direction="maximize", pruner=pruner)

    start_time = time.time()

    study.optimize(objective, n_trials=100, timeout=36000) # 100 trials or 10h

    elapsed_time = time.time() - start_time

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_params = trial.params
    best_params["input_dim"] = 2248  # Add fixed parameters
    best_params["hidden_dim"] = trial.params["hidden_dim"]
    best_params["output_dim"] = trial.params["output_dim"]
    best_params["t_hidden_dim"] = trial.params["t_hidden_dim"]
    best_params["num_layers"] = trial.params["num_layers"]
    best_params["t_num_layers"] =  trial.params["t_num_layers"]
    best_params["num_heads"] = trial.params["num_heads"]
    best_params["opt_lr"] = trial.params["opt_lr"]
    best_params["weight_decay"] = trial.params["weight_decay"]
    best_params["batch_size"] = trial.params["batch_size"]
    best_params["p"] = trial.params["p"]

    # config path
    now = datetime.now()
    config_path = os.path.join("config", now.strftime("optuna_config_%Y-%m-%d_%HUhr") + ".json")

    # Write the best parameters to a JSON file
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)

    print("Best parameters saved")
