"""
PMAS - CUDA Optimized RUL Estimator
BiLSTM Remaining Useful Life Model (PyTorch)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
import joblib
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# DEVICE (CUDA)
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(MODEL_DIR, "data", "Predictive_Maintenance_Synthetic_Data.csv")
SAVE_PATH = os.path.join(MODEL_DIR, "saved_models")
os.makedirs(SAVE_PATH, exist_ok=True)

SEQ_LEN = 48
BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
RUL_CAP = 500
TEST_SPLIT = 0.2

FEATURES = [
    "process_temperature", "air_temperature", "vibration",
    "torque", "rpm", "current", "operating_hours",
    "time_since_last_maintenance", "idle_duration", "power_consumption"
]

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

print("\nRows:", len(df))
print("Machines:", df["machine_id"].nunique())

# ─────────────────────────────────────────
# COMPUTE RUL
# ─────────────────────────────────────────
def compute_rul(group):
    group = group.reset_index(drop=True)
    fail_pos = group.index[group["machine_failure"] == 1].tolist()
    n = len(group)
    rul_vals = np.zeros(n, dtype=float)

    if len(fail_pos) == 0:
        rul_vals = np.arange(n, 0, -1, dtype=float)
    else:
        prev = 0
        for fp in fail_pos:
            for i in range(prev, fp + 1):
                rul_vals[i] = fp - i
            prev = fp + 1
        for i in range(fail_pos[-1] + 1, n):
            rul_vals[i] = n - i

    group["RUL"] = np.clip(rul_vals, 0, RUL_CAP)
    return group

df = df.groupby("machine_id", group_keys=False).apply(compute_rul)
df = df.reset_index()

if "machine_id" not in df.columns:
    df = df.rename(columns={"level_0": "machine_id"})

# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
df["maintenance_enc"] = (df["last_maintenance_Type"] == "corrective").astype(float)
FEATURES_FINAL = FEATURES + ["maintenance_enc"]

feat_scaler = MinMaxScaler()
rul_scaler = MinMaxScaler()

df[FEATURES_FINAL] = feat_scaler.fit_transform(df[FEATURES_FINAL])
df["RUL_scaled"] = rul_scaler.fit_transform(df[["RUL"]])

joblib.dump(feat_scaler, f"{SAVE_PATH}/feat_scaler.pkl")
joblib.dump(rul_scaler, f"{SAVE_PATH}/rul_scaler.pkl")

# ─────────────────────────────────────────
# SEQUENCE BUILDER
# ─────────────────────────────────────────
def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-1])
        y.append(data[i+seq_len-1, -1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train_all, y_train_all = [], []
X_test_all, y_test_all = [], []

for machine, grp in df.groupby("machine_id"):
    arr = grp[FEATURES_FINAL + ["RUL_scaled"]].values
    split = int(len(arr) * (1 - TEST_SPLIT))
    train_arr = arr[:split]
    test_arr = arr[split:]

    Xtr, ytr = make_sequences(train_arr, SEQ_LEN)
    Xte, yte = make_sequences(test_arr, SEQ_LEN)

    X_train_all.append(Xtr)
    y_train_all.append(ytr)
    X_test_all.append(Xte)
    y_test_all.append(yte)

X_train = np.concatenate(X_train_all)
y_train = np.concatenate(y_train_all)
X_test = np.concatenate(X_test_all)
y_test = np.concatenate(y_test_all)

# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────
class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    RULDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)

test_loader = DataLoader(
    RULDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
class BiLSTM_RUL(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

model = BiLSTM_RUL(X_train.shape[2]).to(device)

# ─────────────────────────────────────────
# TRAINING SETUP
# ─────────────────────────────────────────
criterion = nn.HuberLoss(delta=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler()

# ─────────────────────────────────────────
# TRAIN LOOP (FIXED TQDM)
# ─────────────────────────────────────────
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")

    for Xb, yb in train_loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            preds = model(Xb)
            loss = criterion(preds, yb)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.update(1)

    pbar.close()

    epoch_loss = total_loss / len(train_loader)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), f"{SAVE_PATH}/rul_model.pth")
        print(f">> Best model saved at epoch: {epoch} - loss: {epoch_loss}")
        
    scheduler.step()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# ─────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────
model.eval()
preds_all = []
true_all = []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        preds = model(Xb)
        preds_all.append(preds.cpu().numpy())
        true_all.append(yb.numpy())

y_pred_scaled = np.concatenate(preds_all).flatten()
y_true_scaled = np.concatenate(true_all).flatten()

y_pred = rul_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
y_true = rul_scaler.inverse_transform(y_true_scaled.reshape(-1,1)).flatten()

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n==== RESULTS ====")
print("MAE :", round(mae,2), "hours")
print("RMSE:", round(rmse,2), "hours")
print("R2  :", round(r2,4))

config = {
    "seq_len": SEQ_LEN,
    "features": FEATURES_FINAL,
    "rul_cap": RUL_CAP,
    "input_dim": len(FEATURES_FINAL)
}

joblib.dump(config, f"{SAVE_PATH}/model_config.pkl")