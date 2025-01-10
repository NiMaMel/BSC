import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.fst_model import Model
from data.dataset_modul import Dataset
from utils.preprocessing import preprocessing
from utils.utils import shutdown_pc, formatResults, sendEmail, train_rf, train_model, combined_metrics, eval_rf, mean_scores

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

config_path = 'configs/fst_config.json' #  'configs/hpSearch_config.json'
with open(config_path) as json_file:
    model_config = json.load(json_file)

# only uncomment for test purposes
#model_config["encodeLabels"] = False

# 1. Load Dataset and create Tiplet-Df
dataset = pd.read_csv(sider_path)
triplets = [(i,j,row.iloc[j]) for i,row in dataset.iterrows() for j in range(1,len(row))]
triplet_df = pd.DataFrame(triplets, columns=['mol_id', 'target_id', 'label'])

# Experiments
val_scores = pd.DataFrame([])
test_scores = pd.DataFrame([])
bl_scores = pd.DataFrame([])

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
    warmup_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=1, eta_min=1e-6)

    # 4. Dataloader
    BATCH_SIZE = model_config["batch_size"]
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Modeltraining

    # setup folder for tensorboardlogs
    parent_dir = "runs/run_seeds/"

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # initalize writer
    log_dir = os.path.join(parent_dir, f"{datetime.now().strftime('%d-%m-%Y_%Hh-%Mm')}_{seed=}")
    writer = SummaryWriter(log_dir=log_dir)

    # setup savepath for model
    model_parent = "trainedModels/run_seeds/"

    if not os.path.exists(model_parent):
        os.makedirs(model_parent)

    save_path = os.path.join(model_parent, f"model_{seed=}.mdl")

    train_config = {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "max_epochs": 50,
        "patience": 4,
        "log_interval": 100,
        "val_interval": 2,  # set higher to fasten up the process of training
        "save_path": save_path
    }

    train_model(train_config, writer, train_loader, val_loader, device, BATCH_SIZE, useScheduler = False, scheduler = warmup_scheduler)

    # 6. Evaluation

    # load best model
    model.load_state_dict(torch.load(train_config["save_path"]))

    val_score = combined_metrics(model, val_loader, device)
    test_score= combined_metrics(model, test_loader, device)

    val_score.insert(0, "Seed", seed)
    test_score.insert(0, "Seed", seed)

    # 7. Compare to RF Baseline
    y_hats_proba, y_hats_class, true_labels = train_rf(dataset, data, test_triplet, seed, 1000, shuffle=True)
    bl_score = eval_rf(y_hats_class, true_labels)
    bl_score.insert(0, "Seed", seed)

    # update Score-Df's
    val_scores = pd.concat([val_scores, val_score])
    test_scores = pd.concat([test_scores, test_score])
    bl_scores = pd.concat([bl_scores, bl_score])

    print(f"Current Evaluations: Test-AUC:{test_score['AUC']}, Test-D-AUC PR:{test_score['D-AUC PR']}")
    print(f"Experiment {i+1} finished!\n")

print("Experiment done!\n")

# gather configs
usedConfig = {
    'criterion': {
        'class': criterion.__class__.__name__  # Get criterion class name
    },
    'optimizer': {
        'class': optimizer.__class__.__name__,  # Get optimizer class name
        'defaults': optimizer.defaults  # Extract default parameters
    },
    'model_config': model_config,
    'training_params': {k: v for k, v in train_config.items() if k not in ['model','criterion','optimizer']}
    }

avg_val_scores, avg_test_scores, avg_bl_scores = mean_scores(val_scores, test_scores, bl_scores, "results.csv", usedConfig)
print(f"{avg_val_scores=}\n{avg_test_scores=}\n{avg_bl_scores=}")

# send scores to email
email_body = formatResults(avg_val_scores,avg_test_scores,avg_bl_scores)

load_dotenv("email.env")

email_sender = os.getenv("EMAIL_SENDER")
email_password = os.getenv("EMAIL_PASSWORD")
email_receiver = os.getenv("EMAIL_RECEIVER")

sendEmail(email_body, email_receiver, email_sender, email_password)

# when done shut down pc with 60s delay
shutdown_pc()