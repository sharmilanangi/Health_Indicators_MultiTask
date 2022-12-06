import datasets
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from torch.utils.tensorboard import SummaryWriter

import scipy
from scipy import stats

import argparse
from models import SingleTaskNet, MultiTaskNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE IS ... ", device)

def collate_fn(data):
    x, y_df = data
    x_inp = x.to(device)
    y_bmi = torch.tensor(y_df["Mean_BMI"].values, dtype=torch.float32, device=device)
    y_cmr = torch.tensor(
        y_df["Under5_Mortality_Rate"].values, dtype=torch.float32, device=device
    )
    return x_inp, y_bmi, y_cmr

def load_dataset(batch_size):

    with open("data/train_top_feat_tensor.pt", "rb") as f:
        train_proj = torch.load(f)
    with open("data/dev_top_feat_tensor.pt", "rb") as f:
        dev_proj = torch.load(f)
    with open("data/test_top_feat_tensor.pt", "rb") as f:
        test_proj = torch.load(f)

    train_labels = pd.read_parquet(
        "data/train_20221130.parquet.gzip", columns=["Mean_BMI", "Under5_Mortality_Rate"]
    )
    dev_labels = pd.read_parquet(
        "data/dev_20221130.parquet.gzip", columns=["Mean_BMI", "Under5_Mortality_Rate"]
    )
    test_labels = pd.read_parquet(
        "data/test_20221130.parquet.gzip", columns=["Mean_BMI", "Under5_Mortality_Rate"]
    )

    print("DATA LOADED")

    train_dataloader = DataLoader(
    TensorDataset(*collate_fn((train_proj, train_labels))), batch_size=batch_size
    )
    dev_dataloader = DataLoader(
        TensorDataset(*collate_fn((dev_proj, dev_labels))), batch_size=batch_size
    )
    test_dataloader = DataLoader(
        TensorDataset(*collate_fn((test_proj, test_labels))), batch_size=batch_size
    )

    return train_dataloader, dev_dataloader, test_dataloader




def masked_mse(output, target):
    mse_loss = nn.MSELoss()
    mask = torch.isnan(target)
    target = torch.where(mask, 0.0, target)
    output = torch.where(mask, 0.0, output)
    return mse_loss(target, output)



# def r2_loss(output, target):
#     target_mean = torch.nanmean(target)
#     ss_tot = torch.nansum(((target - target_mean)) ** 2)
#     ss_res = torch.nansum(((target - output)) ** 2)
#     r2 = 1 - ss_res / ss_tot
#     return r2

def get_r2_loss(all_preds, all_y):
    bad = ~np.logical_or(np.isnan(all_preds), np.isnan(all_y))
    all_preds_filtered  = np.compress(bad, all_preds)  
    all_y_filtered = np.compress(bad, all_y) 

    r2, _ = stats.pearsonr(all_preds_filtered, all_y_filtered)
    r2 = r2 ** 2

    return r2

def evaluate_model(model, dataloader,loss_fn, bmi_weight, cmr_weight):
    mse_bmi = []
    mse_cmr = []

    all_y_bmi = []
    all_y_cmr = []

    preds_y_bmi = []
    preds_y_cmr = []

    mse_loss = []

    for idx, batch in enumerate(dataloader):
        x, y_bmi, y_cmr = batch
        with torch.no_grad():
            out_bmi, out_cmr = model(x)
            out_bmi, out_cmr = out_bmi.squeeze(), out_cmr.squeeze()

            loss_bmi = loss_fn(out_bmi, y_bmi)
            loss_cmr = loss_fn(out_cmr, y_cmr)

            loss = bmi_weight * loss_bmi + cmr_weight* loss_cmr

            all_y_bmi.append(y_bmi.cpu().numpy())
            all_y_cmr.append(y_cmr.cpu().numpy())

            preds_y_bmi.append(out_bmi.detach().cpu().numpy())
            preds_y_cmr.append(out_cmr.detach().cpu().numpy())

            mse_loss.append(loss.item())
            mse_bmi.append(loss_bmi.item())
            mse_cmr.append(loss_cmr.item())

    
    mse_loss_avg = np.array(mse_loss).mean()
    mse_bmi_avg = np.array(mse_bmi).mean()
    mse_cmr_avg = np.array(mse_cmr).mean()

    preds_y_cmr = np.concatenate(preds_y_cmr, axis=0)
    preds_y_bmi = np.concatenate(preds_y_bmi, axis=0)

    all_y_cmr = np.concatenate(all_y_cmr, axis=0)
    all_y_bmi = np.concatenate(all_y_bmi, axis=0)


    r2_cmr = get_r2_loss(preds_y_cmr, all_y_cmr)
    r2_bmi = get_r2_loss(preds_y_bmi, all_y_bmi)

    return mse_bmi_avg, mse_cmr_avg, mse_loss_avg, r2_bmi, r2_cmr


def main(config):
    print(config)
    writer = SummaryWriter(config.logdir)

    epochs = config.epochs
    lr = config.lr
    batch_size = config.batch_size
    bmi_weight = config.bmi_weight
    cmr_weight = config.cmr_weight

    train_dataloader, dev_dataloader, test_dataloader = load_dataset(batch_size)

    print("Model loading")

    shared_layer_sizes=[462, 512, 256]
    task_head_layers=[256, 128, 64, 1]

    model = MultiTaskNet(shared_layer_sizes = shared_layer_sizes, task_head_layers= task_head_layers).to(device)
    loss_fn = masked_mse
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("data loaders ...")

    best_valid_loss = float("inf")
    for e in range(epochs):
        print("Training ... ")
        train_loss = []
        mse_bmi = []
        mse_cmr = []
        idx = 0
        for batch in train_dataloader:
            x, y_bmi, y_cmr = batch
            out_bmi, out_cmr = model(x)
            out_bmi, out_cmr = out_bmi.squeeze(), out_cmr.squeeze()

            loss_bmi = loss_fn(out_bmi, y_bmi)
            loss_cmr = loss_fn(out_cmr, y_cmr)

            loss = bmi_weight * loss_bmi + cmr_weight* loss_cmr

            train_loss.append(loss.item())
            mse_bmi.append(loss_bmi.item())
            mse_cmr.append(loss_cmr.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            idx += 1
            print(f"Epoch - {e}, Batch - {idx}, Loss -  {loss}")

        train_loss_avg = np.array(train_loss).mean()
        train_mse_bmi = np.array(mse_bmi).mean()
        train_mse_cmr = np.array(mse_cmr).mean()

        print(
        f"===========> TRAIN EPOCH {e}, TRAIN BMI MSE {train_mse_bmi} , TRAIN CMR MSE {train_mse_cmr} "
        )
        print("Running Validation")

        (
            dev_mse_bmi_avg,
            dev_mse_cmr_avg,
            dev_mse_loss_avg,
            dev_r2_bmi_avg,
            dev_r2_cmr_avg,
        ) = evaluate_model(model, dev_dataloader,loss_fn, bmi_weight, cmr_weight)
        if (e + 1) % 1 == 0:  ### saving checkpoint for every 5 epochs
            if dev_mse_loss_avg < best_valid_loss:
                best_valid_loss = dev_mse_loss_avg
                print(f"\nBest validation loss: {best_valid_loss}")
                print(f"\nSaving best model for epoch: {e+1}\n")
                torch.save(
                    {
                        "epoch": e + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": dev_mse_loss_avg,
                    },
                    config.model_path,
                )
        
        # import pdb

        # pdb.set_trace()

        writer.add_scalar("training/MSE Loss", train_loss_avg, e)
        writer.add_scalar("training/MSE BMI", train_mse_bmi, e)
        writer.add_scalar("training/MSE CMR", train_mse_cmr, e)

        writer.add_scalar("eval/BMI_MSE", dev_mse_bmi_avg, e)
        writer.add_scalar("eval/BMI_r2", dev_r2_bmi_avg, e)

        writer.add_scalar("eval/CMR_MSE", dev_mse_cmr_avg, e)
        writer.add_scalar("eval/CMR_r2", dev_r2_cmr_avg, e)

        print(
            f"===========> VALIDATION EPOCH {e}, MSE LOSS - {dev_mse_loss_avg}, BMI R2 LOSS - {dev_r2_bmi_avg}, CMR R2 LOSS - {dev_r2_cmr_avg} "
        )


    print("TESTING THE MODEL")

    best_model = MultiTaskNet(shared_layer_sizes = shared_layer_sizes, task_head_layers= task_head_layers).to(device)
    checkpoint = torch.load(config.model_path)
    print("Loading the best model after epoch - ", checkpoint["epoch"])
    best_model.load_state_dict(checkpoint['model_state_dict'])

    (
        test_mse_bmi_avg,
        test_mse_cmr_avg,
        test_mse_loss_avg,
        test_r2_bmi_avg,
        test_r2_cmr_avg,
    ) = evaluate_model(model, test_dataloader,loss_fn, bmi_weight, cmr_weight)
    print(
        f"===========> TEST RESULTS - TOTAL MSE LOSS - {test_mse_loss_avg}, BMI R2 LOSS - {test_r2_bmi_avg}, CMR R2 LOSS - {test_r2_cmr_avg} "
    )



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    # parser.add_argument('--mode', type=str, default="bmi", help='Choose from bmi and cmr ')
    parser.add_argument('--bmi_weight', type=float, default=0.5)
    parser.add_argument('--cmr_weight', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--logdir', type=str,
                        default='logdir/temp_log')
    parser.add_argument('--model_path', type=str,
                        default='outputs/temp.pth')
    main(parser.parse_args())

# python train_refactor_multitask_top_feat.py --epochs 200 --bmi_weight 0.5 --cmr_weight 0.5 --logdir logdir/multi_bmi_0.5_cmr_0.5_deep_200 --model_path outputs/multi_bmi_0.5_cmr_0.5_deep_200.pth