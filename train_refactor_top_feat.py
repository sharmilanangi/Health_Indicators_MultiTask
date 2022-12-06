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
from models import SingleTaskNet


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
        "data/train_20221130.parquet.gzip",
        columns=["Mean_BMI", "Under5_Mortality_Rate"],
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


def r2_loss(output, target):
    target_mean = torch.nanmean(target)
    ss_tot = torch.nansum(((target - target_mean)) ** 2)
    ss_res = torch.nansum(((target - output)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def evaluate_model(model, dataloader, loss_fn, mode):

    mse_loss = []

    all_preds = []
    all_y = []

    for idx, batch in enumerate(dataloader):
        x, y_bmi, y_cmr = batch
        if mode == "cmr":
            y = y_cmr
        elif mode == "bmi":
            y = y_bmi
        with torch.no_grad():
            outs = model(x).squeeze()
            loss = loss_fn(outs, y)

            mse_loss.append(loss.item())

            all_y.append(y.cpu().numpy())
            preds_numpy = outs.detach().cpu().numpy()
            all_preds.append(preds_numpy)

    all_preds = np.concatenate(all_preds, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    mse_loss_avg = np.array(mse_loss).mean()

    bad = ~np.logical_or(np.isnan(all_preds), np.isnan(all_y))
    all_preds_filtered = np.compress(bad, all_preds)
    all_y_filtered = np.compress(bad, all_y)

    r2, _ = stats.pearsonr(all_preds_filtered, all_y_filtered)
    r2 = r2**2

    return mse_loss_avg, r2


def main(config):
    print(config)
    writer = SummaryWriter(config.logdir)

    epochs = config.epochs
    lr = config.lr
    batch_size = config.batch_size

    train_dataloader, dev_dataloader, test_dataloader = load_dataset(batch_size)

    print("Model loading")

    layer_sizes = [462, 512, 256, 128]

    model = SingleTaskNet(layer_sizes=layer_sizes).to(device)
    loss_fn = masked_mse
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mode = config.mode

    print("data loaders ...")

    best_valid_loss = float("inf")
    for e in range(epochs):
        print("Training ... ")
        train_loss = []
        idx = 0
        for batch in train_dataloader:
            x, y_bmi, y_cmr = batch
            if mode == "cmr":
                y = y_cmr
            elif mode == "bmi":
                y = y_bmi
            outs = model(x).squeeze()
            loss = loss_fn(outs, y)

            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            idx += 1
            print(f"Epoch - {e}, Batch - {idx}, Loss -  {loss}")

        loss_per_epoch = np.array(train_loss).mean()

        print(f"===========> TRAIN EPOCH {e}, TRAIN LOSS PER EPOCH {loss_per_epoch} ")

        print("Running Validation")

        dev_loss_per_epoch, dev_r2_loss_per_epoch = evaluate_model(
            model, dev_dataloader, loss_fn, mode
        )

        if (e + 1) % 1 == 0:  ### saving checkpoint for every 5 epochs
            if dev_loss_per_epoch < best_valid_loss:
                best_valid_loss = dev_loss_per_epoch
                print(f"\nBest validation loss: {best_valid_loss}")
                print(f"\nSaving best model for epoch: {e+1}\n")
                torch.save(
                    {
                        "epoch": e + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": dev_loss_per_epoch,
                    },
                    config.model_path,
                )

        writer.add_scalar("training/MSE Loss", loss_per_epoch, e)
        writer.add_scalar("eval/MSE Loss", dev_loss_per_epoch, e)
        writer.add_scalar("eval/R2 SCORE", dev_r2_loss_per_epoch, e)

        print(
            f"===========> VALIDATION EPOCH {e}, MSE LOSS - {dev_loss_per_epoch}, R2 LOSS - {dev_r2_loss_per_epoch} "
        )

    print("TESTING THE MODEL")

    best_model = SingleTaskNet(layer_sizes=layer_sizes).to(device)
    checkpoint = torch.load(config.model_path)
    print("Loading the best model after epoch - ", checkpoint["epoch"])
    best_model.load_state_dict(checkpoint["model_state_dict"])

    test_loss_per_epoch, test_r2_loss_per_epoch = evaluate_model(
        best_model, test_dataloader, loss_fn, mode
    )
    print(
        f"===========> FINAL TEST, MSE LOSS - {test_loss_per_epoch}, R2 LOSS - {test_r2_loss_per_epoch} "
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--mode", type=str, default="bmi", help="Choose from bmi and cmr "
    )
    # parser.add_argument('--bmi_weight', type=float, default=0.5)
    # parser.add_argument('--cmr_weight', type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="logdir/temp_log")
    parser.add_argument("--model_path", type=str, default="outputs/temp.pth")
    main(parser.parse_args())

# python train_refactor_top_feat.py --epochs 2 --mode cmr --logdir logdir/cmr_refactor_tmp --model_path outputs/cmr_refactor_tmp.pth

# python train_refactor_top_feat.py --epochs 200 --mode bmi --logdir logdir/bmi_top_feat_200_lr_e5 --model_path outputs/bmi_top_feat_200_lr_e5.pth
