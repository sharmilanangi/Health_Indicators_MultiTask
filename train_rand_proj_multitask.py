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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE IS ... ", device)


class MultiTaskNet(nn.Module):
    def __init__(self, embed_dim=11348, layer_sizes=[2048, 500, 2048, 500]):
        super().__init__()

        self.embedding_dim = embed_dim

        self.mlp_net = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),  ## 96x64
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),  ## 64x1
            nn.ReLU(),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
        )

        self.last_layer_bmi = nn.Linear(
            layer_sizes[3], 1
        )  ## change if we need classification or softmax
        self.last_layer_cmr = nn.Linear(
            layer_sizes[3], 1
        )  ## change if we need classification or softmax

    def forward(self, x):
        x = self.mlp_net(x)

        out_bmi = self.last_layer_bmi(x)
        out_cmr = self.last_layer_cmr(x)

        return out_bmi, out_cmr


with open("data/train_random_proj.pt", "rb") as f:
    train_proj = torch.load(f)
with open("data/dev_random_proj.pt", "rb") as f:
    dev_proj = torch.load(f)
with open("data/test_random_proj.pt", "rb") as f:
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


def collate_fn(data):
    x, y_df = data
    x_inp = x.to(device)
    y_bmi = torch.tensor(y_df["Mean_BMI"].values, dtype=torch.float32, device=device)
    y_cmr = torch.tensor(
        y_df["Under5_Mortality_Rate"].values, dtype=torch.float32, device=device
    )
    return x_inp, y_bmi, y_cmr


epochs = 200
lr = 1e-5
batch_size = 256

print("data loaders ...")


train_dataloader = DataLoader(
    TensorDataset(*collate_fn((train_proj, train_labels))), batch_size=batch_size
)
dev_dataloader = DataLoader(
    TensorDataset(*collate_fn((dev_proj, dev_labels))), batch_size=batch_size
)
test_dataloader = DataLoader(
    TensorDataset(*collate_fn((test_proj, test_labels))), batch_size=batch_size
)


def masked_mse(output, target):
    mse_loss = nn.MSELoss()
    mask = torch.isnan(target)
    target = torch.where(mask, 0.0, target)
    output = torch.where(mask, 0.0, output)
    return mse_loss(target, output)


print("Model loading")
model = MultiTaskNet().to(device)
loss_fn = masked_mse
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


def r2_loss(output, target):
    target_mean = torch.nanmean(target)
    ss_tot = torch.nansum(((target - target_mean)) ** 2)
    ss_res = torch.nansum(((target - output)) ** 2)
    r2 = 1 - ss_res / ss_tot

    mask = torch.isnan(target)
    return torch.where(mask, 0.0, r2)


def evaluate_model(model, dataloader):
    mse_bmi = []
    mse_cmr = []

    r2_bmi_vals = []
    r2_cmr_vals = []

    mse_loss = []

    for idx, batch in enumerate(dataloader):
        x, y_bmi, y_cmr = batch
        with torch.no_grad():
            out_bmi, out_cmr = model(x)
            out_bmi, out_cmr = out_bmi.squeeze(), out_cmr.squeeze()

            loss_bmi = loss_fn(out_bmi, y_cmr)
            loss_cmr = loss_fn(out_cmr, y_cmr)

            loss = loss_bmi + loss_cmr

            r2_bmi = r2_loss(out_bmi, y_cmr)
            r2_cmr = r2_loss(out_cmr, y_cmr)

            mse_loss.append(loss.item())
            mse_bmi.append(loss_bmi.item())
            mse_cmr.append(loss_cmr.item())

            r2_bmi_vals.append(r2_bmi.item())
            r2_cmr_vals.append(r2_cmr.item())

    mse_loss_avg = np.array(mse_loss).mean()
    mse_bmi_avg = np.array(mse_bmi).mean()
    mse_cmr_avg = np.array(mse_cmr).mean()

    r2_cmr_avg = np.array(r2_cmr_vals).mean()
    r2_bmi_avg = np.array(r2_bmi_vals).mean()

    return mse_bmi_avg, mse_cmr_avg, mse_loss_avg, r2_bmi_avg, r2_cmr_avg


writer = SummaryWriter("logdir/multitask_randproj")
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

        loss_bmi = loss_fn(out_bmi, y_cmr)
        loss_cmr = loss_fn(out_cmr, y_cmr)
        loss = loss_bmi + loss_cmr

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
    ) = evaluate_model(model, dev_dataloader)
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
                "outputs/best_multitask.pth",
            )

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

(
    test_mse_bmi_avg,
    test_mse_cmr_avg,
    test_mse_loss_avg,
    test_r2_bmi_avg,
    test_r2_cmr_avg,
) = evaluate_model(model, test_dataloader)
print(
    f"===========> VALIDATION EPOCH {e}, MSE LOSS - {test_mse_loss_avg}, BMI R2 LOSS - {test_r2_bmi_avg}, CMR R2 LOSS - {test_r2_cmr_avg} "
)
