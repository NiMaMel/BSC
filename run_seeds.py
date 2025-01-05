import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.fst_model import Model
from data.dataset_modul import Dataset
from utils.preprocessing import preprocessing
from utils.utils import shutdown_pc, train_rf, train_model, combined_metrics, eval_rf, mean_scores

# dataset paths
sider_path = "data/datasets/sider.csv"
tox_path = "data/datasets/toxcast_data.csv"

# set hardcoded seed to generate 10 random seeds for the experiments
np.random.seed(42)

# generate seeds for experiments
seeds = np.random.randint(0, 10000, size=10)

# cuda setting
if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    device = torch.device('cuda:%d' % device_id)
else:
    device = torch.device('cpu')
print(f"device set:{torch.cuda.get_device_name(device_id)}\n")

#default config -> best results
with open('configs/fst_config.json') as json_file:
    model_config = json.load(json_file)

#with open('default_config_no_wdecay.json') as json_file:
#    model_config = json.load(json_file)

# hp search config
#with open('configs/hpSearch_config.json') as json_file:
#    model_config = json.load(json_file)

# 1. Load Dataset and create Tiplet-Df
dataset = pd.read_csv(sider_path)
triplets = [(i,j,row.iloc[j]) for i,row in dataset.iterrows() for j in range(1,len(row))]
triplet_df = pd.DataFrame(triplets, columns=['mol_id', 'target_id', 'label'])

# Experiments
val_scores = pd.DataFrame([])
test_scores = pd.DataFrame([])
rf_scores = pd.DataFrame([])

for i, seed in enumerate(seeds):
    print(f"Running experiment {i+1} with seed {seed}")

    # set current seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. Preprocessing & Datasplit
    data, dataset_run, train_triplet, val_triplet, test_triplet = preprocessing(dataset, triplet_df, seed)

    train_set = Dataset(data, dataset_run, train_triplet, supp=8, seed=seed)
    val_set = Dataset(data, dataset_run, val_triplet, supp=8, train=False, seed=seed)
    test_set = Dataset(data, dataset_run, test_triplet, supp=8, train=False, seed=seed)

    # 3. Initialize Model
    model_params = {k: v for k, v in model_config.items() if k not in ['opt_lr', 'weight_decay']}
    model = Model(**model_params) 
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=model_config["opt_lr"], weight_decay=model_config["weight_decay"])

    # 4. Dataloader
    BATCH_SIZE = model_config["batch_size"]
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Modeltraining

    train_config = {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "max_epochs": 50,
        "patience": 4,
        "log_interval": 100,
        "val_interval": 1,  # set higher to fasten up the process of training
        "save_path": f"{os.path.join('trainedModels', f'model_{seed}_run{i+1}')}.mdl"
    }

    log_dir = f"runs/run_seeds_{datetime.now().strftime('%d-%m-%Y_%Hh-%Mm-%Ss')}_{seed}"
    writer = SummaryWriter(log_dir=log_dir)
    
    train_model(train_config, writer, train_loader, val_loader, device, BATCH_SIZE)

    # 6. Evaluation

    # load best model
    model.load_state_dict(torch.load(train_config["save_path"]))

    val_score = combined_metrics(model, val_loader, device)
    test_score= combined_metrics(model, test_loader, device)

    val_score.insert(0, "Seed", seed)
    test_score.insert(0, "Seed", seed)

    # 7. Compare to RF Baseline
    y_hats_proba, y_hats_class, true_labels = train_rf(dataset, data, test_triplet, seed, 1000, shuffle=True)
    rf_score = eval_rf(y_hats_class, true_labels)
    rf_score.insert(0, "Seed", seed)

    # update Score-Df's
    val_scores = pd.concat([val_scores, val_score])
    test_scores = pd.concat([test_scores, test_score])
    rf_scores = pd.concat([rf_scores, rf_score])

    print(f"Current Evaluations: Test-AUC:{test_score['AUC']}, Test-D-AUC PR:{test_score['D-AUC PR']}")
    print(f"Experiment {i+1} finished!\n")

print("Experiment done!\n")

avg_val_scores, avg_test_scores, avg_rf_scores = mean_scores(val_scores, test_scores, rf_scores,"results_wOLabelEncoding.csv")
print(f"{avg_val_scores=}\n{avg_test_scores=}\n{avg_rf_scores=}")

# when done shot down pc with 60s delay
#shutdown_pc()