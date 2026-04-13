"""
Single Run Training - DySurv

Date Last updated: XXXX
Author: Munib & Kaca
Please contact via _
"""
import json

from src.data_processing.data_loader import data_loader
import src.models.model_utils as model_utils
from src.results.main import evaluate
import src.visualisation.main as vis_main

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import numba
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from src.models.survival.metrics import concordance_td

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
from torch import nn
import torch.nn.functional as F
import torchtuples as tt # Some useful functions

from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv
from torch import Tensor

# Model Utils
from src.models.survival.model_utils import DySurv
from src.models.survival.model_utils import Loss
from src.models.survival.model_utils import adaptive_pos_weight
from src.models.survival.model_utils import get_exp_dir
from sklearn.manifold import TSNE

# Visualisation Utils
from src.models.survival.visualisation_utils import plot_res, plot_predict_surv, eval_surv, get_vis_res_dir

import seaborn as sn
sn.set_theme(style="white", palette="rocket_r")

import random
import pickle
import gc

def main():
    
    seed = 1348 #(1348, 3642, 6535, 6886, 7417)

    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)

    "Change directory."
    os.chdir('/Users/katarina/CODE/camelot-icml/')

    "Data Loading."
    with open("src/training/data_config1.json", "r") as f:
        data_config = json.load(f)
        f.close()
    "Training Loading."
    with open("src/models/survival/training_config.json", "r") as f:
        train_config = json.load(f)
        f.close()
    "Model Loading."
    with open("src/models/survival/model_config.json", "r") as f:
        model_config = json.load(f)
        f.close()

    data_config['seed'] = seed
    data_config['random_state'] = seed

    with open(f'data_info_surv_{seed}.pkl', 'rb') as file: 
        data_info = pickle.load(file)

    data_config['seed'] = seed
    data_config['random_state'] = seed

    # Define number of survival curve output units / discrete survival
    num_durations = model_config["num_durations"]
    labtrans = LogisticHazard.label_transform(num_durations)

    ################### Extract time-to-event ###########################
    # Prepares the data in the form [durations, mortality]
    train_data = data_info["X"][0]
    val_data = data_info["X"][1]
    test_data = data_info["X"][2]

    y_train_surv = data_info["y"][0][:,-2:]
    y_val_surv = data_info["y"][1][:,-2:]
    y_test_surv = data_info["y"][2][:,-2:]

    y_train_surv[:,1] = y_train_surv[:,1]
    y_val_surv[:,1] = y_val_surv[:,1]
    y_test_surv[:,1] = y_test_surv[:,1]

    #train_data = train_data[:,:-1,:]
    #val_data = val_data[:,:-1,:]
    #test_data = test_data[:,:-1,:]

    # Fit transform? in y_train_surv [event, duration], input should be [duration, event]
    y_train_surv = labtrans.fit_transform(y_train_surv[:,1], y_train_surv[:,0])
    y_val_surv = labtrans.transform(y_val_surv[:,1], y_val_surv[:,0])

    ##################### Data type check and stuff ###################

    # data needs to be of type  
    x_train = train_data.astype('float32')
    x_val = val_data.astype('float32')
    x_test = test_data.astype('float32')

    # Create place holder, [num_patients, num_feat, 1]
    train_target = np.zeros((x_train.shape[0], x_train.shape[1], 1))
    val_target = np.zeros((x_val.shape[0], x_val.shape[1], 1))

    # Reshape train_target ie [duration, event], create target filled with durations 
    train_target[:, :, -1] = y_train_surv[1].reshape(-1, 1)
    val_target[:, :, -1] = y_val_surv[1].reshape(-1, 1)

    # Append this new column?
    x_train = np.append(x_train, train_target, axis=2)
    x_val = np.append(x_val, val_target, axis=2)

    # Put together [x_train, ([duration, event], x_train)]
    train = tt.tuplefy(x_train, (y_train_surv, x_train))
    val = tt.tuplefy(x_val, (y_val_surv, x_val))

    # Extract on test data
    durations_test = y_test_surv[:,1]
    events_test = y_test_surv[:,0]

    ##################### Gets data properties ##################
    # Number of input features, the -1 is because of the durations that are added
    in_features = x_train.shape[2]-1

    # Number of encoded features
    encoded_features = model_config["encoded_features"] # use 20 latent factors

    # Number of output point to predict for, let's say 10
    out_features = labtrans.out_features # how many discrete time points to predict for (10 here)

    # Define how long our sequence is 
    seq_len = x_train.shape[1]

    # Build the model
    net = DySurv(in_features, encoded_features, out_features, seq_len)

    # Constructing the Loss — pos_weight adapts to the dataset event rate
    pw = adaptive_pos_weight(y_train_surv[1])
    print(f"  pos_weight={pw:.1f} (event_rate={(y_train_surv[1] > 0).mean():.3f})")
    loss = Loss([model_config["loss_surv"], model_config["loss_ae"], model_config["loss_kd"]], pos_weight=pw)

    # Wrapper model for the input, meaning it would have only the nll loss if it hadn't been defined. Also, it allows for surv.predictions etc.
    model = LogisticHazard(net, tt.optim.Adam(train_config["lr"]), duration_index=labtrans.cuts, loss=loss) # wrapper

    # Define the metrics to be used
    metrics = dict(
        loss_surv = Loss([1, 0, 0]),
        loss_ae = Loss([0, 1, 0]),
        loss_kd = Loss([0, 0, 1])
    )
    #callbacks = [tt.cb.EarlyStopping(min_delta=0.0001, patience=3)]
    callbacks = []

    # Model train
    log = model.fit(*train, batch_size = train_config["bs"], epochs = train_config["epochs"], callbacks = callbacks, verbose = 2, val_data=val, metrics=metrics, num_workers=0)
    res = log.to_pandas()

    print("Analysis Complete.")

    ############### Plotting and Saving Results #################

    print("Plotting results.")
    vis_dir, res_dir = get_vis_res_dir(data_load_config=data_config, model_name=model_config["model_name"])

    plot_res(results = res, save_fd=vis_dir)

    surv = plot_predict_surv(data=x_test, surv_times=durations_test, model=model, save_fd=vis_dir)

    eval_surv(surv=surv, durations=durations_test, events=events_test, save_fd=vis_dir, save_res=res_dir)

    ################### Save Params ############################
    model_config['random_state'] = seed
    exp_dir = get_exp_dir(data_load_config=data_config, model_name=model_config["model_name"])
    
    with open(res_dir+'train_config.json', 'w') as file:
        json.dump(train_config, file, indent=4)
    with open(res_dir+'data_config1.json', 'w') as file:
        json.dump(data_config, file, indent=4)
    with open(res_dir+'model_config.json', 'w') as file:
        json.dump(model_config, file, indent=4)
    
    surv_train = model.interpolate(10).predict_surv_df(train_data.astype('float32'))
    y_train_surv = data_info["y"][0][:,-2:]
    for col in surv_train.columns:
        cl_val = surv_train.iloc[:,col][surv_train.iloc[:,col].index <=  y_train_surv[col, 1]].iloc[-1]
        surv_train.loc[surv_train.index >=  y_train_surv[col, 1], surv_train.columns[col]] = cl_val

    surv_train.to_csv(res_dir + 'surv_train.csv', index=True, header=True)

    surv_val = model.interpolate(10).predict_surv_df(val_data.astype('float32'))
    surv_val.to_csv(res_dir + 'surv_val.csv', index=True, header=True)

    surv.to_csv(res_dir + 'surv_test.csv', index=True, header=True)


    print("Parameters saved.")

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
