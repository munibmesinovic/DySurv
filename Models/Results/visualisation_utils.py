import matplotlib.pyplot as plt
import os
import json
import numpy as np

from pycox.evaluation import EvalSurv
from src.models.survival.metrics import concordance_td, idx_at_times
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_recall_curve, auc


def get_vis_res_dir(data_load_config=None, model_name: str = 'DySurv'):

    # Standardise model_name
    if "dysurv" in model_name.lower():
        model_name = "DySurv"
    elif "ddh" in model_name.lower():
        model_name = "DynamicDeephit"
    elif "deephit" in model_name.lower():
        model_name = "Deephit"
    else:
        model_name = "Test"

    # Get save path
    data_name = data_load_config["data_name"]
    run_num = 1
    SAVE_FD = f"visualisations/{data_name}/{model_name}/results_{data_load_config['seed']}/"

    if not os.path.exists(SAVE_FD):
        os.makedirs(SAVE_FD)
    while os.path.exists(SAVE_FD + f"run{run_num}/"):
        run_num += 1

    save_fd = SAVE_FD + f"run{run_num}/"
    if not os.path.exists(save_fd):
        os.makedirs(save_fd)

    SAVE_FD = f"results/{data_name}/{model_name}/results_{data_load_config['seed']}/"
    if not os.path.exists(SAVE_FD):
        os.makedirs(SAVE_FD)
    res_dir = SAVE_FD + f"run{run_num}/"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    return save_fd, res_dir

def plot_res(results=None, save_fd=None):
    "Plotting Loss Curves for the DySurv"

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 18), sharex='all')

    titles = ["Train vs Validation Loss", "Train vs Validation Loss (Surv)", 
            "Train vs Validation Loss (AE)", "Train vs Validation Loss (KD)"]
    data_keys = [('train_loss', 'val_loss'), ('train_loss_surv', 'val_loss_surv'),
                ('train_loss_ae', 'val_loss_ae'), ('train_loss_kd', 'val_loss_kd')]

    for ax, title, (train_key, val_key) in zip(axes.flatten(), titles, data_keys):
        if train_key in results.keys():
            ax.plot(results[train_key], label=f'Train {title.split()[3]}')
            ax.plot(results[val_key], label=f'Validation {title.split()[3]}')
            ax.set(title=title, xlabel='Epoch', ylabel='Loss')
            ax.legend()

    fig.tight_layout()

    fig.savefig(save_fd + "losses.png", dpi=200)
    plt.close(fig)


def plot_predict_surv(data=None, surv_times=None, model=None, save_fd=None):
    "Plotting Survival Predictions for the DySurv model"
    
    surv = model.interpolate(10).predict_surv_df(data)
    surv1 = surv.copy()

    indices = [0, 1, 2, 26]
    
    plt.figure()
    for i in indices:
        surv.iloc[:, i].plot(drawstyle='steps-post') 
    
    plt.ylabel('S(t | x)')
    plt.xlabel('Time')
    plt.legend(['Censored, 45h', 'Censored, 48h', 'Censored, 24h', 'Died, 25h'])

    plt.savefig(save_fd + "predictions_surv_original.png", dpi=200)
    plt.close()

    plt.figure()

    for i in indices:
        cl_val = surv.iloc[:,i][surv.iloc[:,i].index <= surv_times[i]].iloc[-1]
        surv1.loc[surv1.index >= surv_times[i], surv1.columns[i]] = cl_val
        surv1.iloc[:, i].plot(drawstyle='steps-post') 
    
    plt.ylabel('S(t | x)')
    plt.xlabel('Time [h]')
    plt.legend(['Censored, 45h', 'Censored, 48h', 'Censored, 24h', 'Died, 25h'])

    plt.savefig(save_fd + "predictions_surv_adjusted.png", dpi=200)
    plt.close()

    return surv

# censored, long stay, no event, 45 h, 7417
# censored, until the end stay, no event 48h 
# censored, short stay, no event around 24 h
# died, short stay around 25 h 

def eval_surv(surv=None, durations=None, events=None, save_fd=None, save_res=None):
    "Plotting the Evaluated Survival for patients. And getting the Scores."
    
    ev = EvalSurv(surv, durations, events, censor_surv='km')

    time_grid = np.linspace(durations.min(), durations.max(), 100)

    ev.brier_score(time_grid).plot()
    plt.ylabel('Brier score')
    _ = plt.xlabel('Time')
    plt.savefig(save_fd + "brier.png", dpi=200)
    plt.close()

    ev.nbll(time_grid).plot()
    plt.ylabel('NBLL')
    _ = plt.xlabel('Time')
    plt.savefig(save_fd + "NBLL.png", dpi=200)
    plt.close()

    # Administrative censoring
    idxs = np.where(durations < 48)
    durs = durations[idxs]
    evs = events[idxs]
    survs = surv[idxs[0]]
    ev1 = EvalSurv(survs, durs, evs, censor_surv='km')

    res = {}
    res["Integrated Brier Score"] = ev.integrated_brier_score(time_grid)
    res["Integrated NBLL"] = ev.integrated_nbll(time_grid)
    res["Concordance"] = ev.concordance_td()
    #concordance_td(durations = durations, events = events, surv = surv.values, surv_idx = idx_at_times(surv.index.values, durations, 'post'), method='adj_antolini')
    res["Integrated Brier Score - non-admin"] = ev1.integrated_brier_score(time_grid)
    res["Integrated NBLL - non-admin"] = ev1.integrated_nbll(time_grid)
    res["Concordance - non-admin"] = ev1.concordance_td()
    #oncordance_td(durations = durs, events = evs, surv = survs.values, surv_idx = idx_at_times(survs.index.values, durs, 'post'), method='adj_antolini')

    print(f"Printing scores:")
    print(f"Integrated Brier Score: {res['Integrated Brier Score']}")
    print(f"Integrate NBLL        : {res['Integrated NBLL']}")
    print(f"Concordance           : {res['Concordance']}")

    # Custom accuracy
    y_pred = []

    for pat in surv.columns:
        y_pred.append( surv.iloc[:,pat][surv.iloc[:,pat].index <= durations[pat]].iloc[-1] )
    y_pred = [1.-var for var in y_pred]

    # Compute prediction accuracy
    res['f1'] = f1_score(events, [1 if p < 0.5 else 0 for p in y_pred])
    res['recall'] = recall_score(events, [1 if p < 0.5 else 0 for p in y_pred])
    res['auroc'] = roc_auc_score(events, y_pred)
    precision, recall_vals, _ = precision_recall_curve(events, y_pred)
    res['auprc'] = auc(recall_vals, precision)

    print(f"F1                    : {res['f1']}")
    print(f"Recall                : {res['recall']}")
    print(f"AUROC                 : {res['auroc']}")
    print(f"AUPRC                 : {res['auprc']}")

    with open(save_res+'scores.json', 'w') as file:
        json.dump(res, file, indent=4)
