import datasets
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE IS ... ", device)


class MultiTaskNet(nn.Module):
    def __init__(self, embed_dim=11348, layer_sizes=[11348, 500, 11348, 500]):
        super().__init__()

        self.embedding_dim = embed_dim

        self.mlp_net = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),  ## 96x64
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),  ## 64x1
            nn.ReLU(),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
        )

        self.last_layer = nn.Linear(
            layer_sizes[3], 1
        )  ## change if we need classification or softmax

    def forward(self, x):
        x = self.mlp_net(x)

        out_x = self.last_layer(x)

        return out_x


dataset = datasets.load_dataset(
    "parquet",
    data_files={
        "train": "./data_processed/train.parquet.gzip",
        "dev": "./data_processed/dev.parquet.gzip",
        "test": "./data_processed/test.parquet.gzip",
    },
)

print("DATA LOADED")

other_keys = [
    "Mean_BMI",
    "Under5_Mortality_Rate",
    "Stunted_Rate",
    "new_ind",
    "key1",
    "key2",
    "key3",
    "DATUM",
    "DHSCC",
    "DHSID_x",
    "DHSREGNA",
    "SOURCE",
    "URBAN_RURA_x",
    "CCFIPS",
    "DHSID_y",
    "URBAN_RURA_y",
    "ADM1NAME",
]


def collate_fn_restructure(data):

    df = pd.DataFrame(
        data,
    )
    all_keys = list(df.keys())
    feat_keys = [i for i in all_keys if i not in other_keys]

    x_inp = torch.tensor(df[feat_keys].values, dtype=torch.float32, device=device)
    y_bmi = torch.tensor(df["Mean_BMI"].values, dtype=torch.float32, device=device)
    y_cmr = torch.tensor(
        df["Under5_Mortality_Rate"].values, dtype=torch.float32, device=device
    )
    return x_inp, y_bmi, y_cmr


epochs = 20
lr = 1e-4
batch_size = 16

print("data loaders ...")
train_dataloader = DataLoader(
    dataset["train"], batch_size=batch_size, collate_fn=collate_fn_restructure
)
dev_dataloader = DataLoader(
    dataset["dev"], batch_size=batch_size, collate_fn=collate_fn_restructure
)
test_dataloader = DataLoader(
    dataset["test"], batch_size=batch_size, collate_fn=collate_fn_restructure
)

print("Model loading")
model = MultiTaskNet().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def evaluate_model(model, dataloader):
    mse_loss = []
    r2_losses = []

    for idx, batch in enumerate(dataloader):
        x, y_bmi, y_cmr = batch
        with torch.no_grad():
            outs = model(x).squeeze()
            loss = loss_fn(outs, y_bmi)
            r2_val_loss = r2_loss(outs, y_bmi)

            mse_loss.append(loss.item())
            r2_losses.append(r2_val_loss.item())

    mse_loss_avg = np.array(mse_loss).mean()
    r2_losses_avg = np.array(r2_losses).mean()

    return mse_loss_avg, r2_losses_avg


writer = SummaryWriter("logdir/bmi_train")
best_valid_loss = float("inf")
for e in range(epochs):
    print("Training ... ")
    train_loss = []
    idx = 0
    for batch in train_dataloader:
        x, y_bmi, y_cmr = batch
        outs = model(x).squeeze()
        loss = loss_fn(outs, y_bmi)

        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        idx += 1
        print(f"Epoch - {e}, Batch - {idx}, Loss -  {loss}")

    loss_per_epoch = np.array(train_loss).mean()

    print(f"===========> TRAIN EPOCH {e}, TRAIN LOSS PER EPOCH {loss_per_epoch} ")

    print("Running Validation")

    dev_loss_per_epoch, dev_r2_loss_per_epoch = evaluate_model(model, dev_dataloader)

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
                "outputs/best_bmi.pth",
            )

    writer.add_scalar("training/MSE Loss", loss_per_epoch, e)
    writer.add_scalar("eval/MSE Loss", dev_loss_per_epoch, e)
    writer.add_scalar("eval/R2 SCORE", dev_r2_loss_per_epoch, e)

    print(
        f"===========> VALIDATION EPOCH {e}, MSE LOSS - {dev_loss_per_epoch}, R2 LOSS - {dev_r2_loss_per_epoch} "
    )

print("TESTING THE MODEL")

test_loss_per_epoch, test_r2_loss_per_epoch = evaluate_model(model, test_dataloader)
print(
    f"===========> FINAL TEST, MSE LOSS - {test_loss_per_epoch}, R2 LOSS - {test_r2_loss_per_epoch} "
)
