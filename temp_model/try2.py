# ============================================================
# ADVANCED Predictive Maintenance Model
# BiLSTM Autoencoder + Feature Engineering
# Optimized for Apple Silicon (MPS)
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1️⃣ DEVICE SETUP (Mac M2 Optimized)
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ============================================================
# 2️⃣ LOAD DATA
# ============================================================

DATA_PATH = "Predictive_Maintenance_Synthetic_Data.csv"
df = pd.read_csv(DATA_PATH)

TARGET_COL = "machine_failure"

# Drop non-numeric columns
df = df.drop(columns=["timestamp", "machine_id", "last_maintenance_Type"])

# ============================================================
# 3️⃣ FEATURE ENGINEERING (VERY IMPORTANT)
# ============================================================

df["power_per_rpm"] = df["power_consumption"] / (df["rpm"] + 1e-6)
df["torque_rpm_ratio"] = df["torque"] / (df["rpm"] + 1e-6)
df["temp_diff"] = df["process_temperature"] - df["air_temperature"]

# ============================================================
# 4️⃣ SPLIT HEALTHY DATA
# ============================================================

healthy_df = df[df[TARGET_COL] == 0].reset_index(drop=True)
failed_df  = df[df[TARGET_COL] == 1].reset_index(drop=True)

print("Healthy samples:", len(healthy_df))
print("Failure samples:", len(failed_df))

# ============================================================
# 5️⃣ SCALING (FIT ONLY ON HEALTHY)
# ============================================================

feature_cols = df.drop(columns=[TARGET_COL]).columns

scaler = StandardScaler()
healthy_scaled = scaler.fit_transform(healthy_df[feature_cols])
full_scaled = scaler.transform(df[feature_cols])

# ============================================================
# 6️⃣ SEQUENCE CREATION (LONGER WINDOW)
# ============================================================

def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

SEQ_LEN = 40

healthy_sequences = create_sequences(healthy_scaled, SEQ_LEN)
full_sequences = create_sequences(full_scaled, SEQ_LEN)

print("Sequence shape:", healthy_sequences.shape)

# ============================================================
# 7️⃣ ADVANCED BiLSTM AUTOENCODER
# ============================================================

class AdvancedLSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super(AdvancedLSTMAE, self).__init__()

        # Encoder (2-layer BiLSTM)
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.fc_latent = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            hidden_dim,
            input_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)

        # Take last layer forward+backward
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)

        latent = self.fc_latent(hidden_cat)

        decoded = self.fc_decode(latent)
        decoded = decoded.unsqueeze(1).repeat(1, x.size(1), 1)

        output, _ = self.decoder(decoded)

        return output

# ============================================================
# 8️⃣ TRAIN SETUP
# ============================================================

model = AdvancedLSTMAE(
    input_dim=healthy_sequences.shape[2]
).to(device)

criterion = nn.SmoothL1Loss()  # Huber Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_dataset = TensorDataset(
    torch.tensor(healthy_sequences, dtype=torch.float32)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

# ============================================================
# 9️⃣ TRAIN MODEL
# ============================================================

EPOCHS = 40

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x = batch[0].to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.6f}")

# ============================================================
# 🔟 RECONSTRUCTION ERRORS
# ============================================================

model.eval()
recon_errors = []

with torch.no_grad():
    for i in range(len(full_sequences)):
        x = torch.tensor(full_sequences[i:i+1], dtype=torch.float32).to(device)
        output = model(x)
        loss = torch.mean((output - x)**2).item()
        recon_errors.append(loss)

recon_errors = np.array(recon_errors)

# ============================================================
# 1️⃣1️⃣ GAUSSIAN THRESHOLDING
# ============================================================

healthy_errors = recon_errors[:len(healthy_sequences)]

mean = healthy_errors.mean()
std = healthy_errors.std()

threshold_healthy = mean + 2 * std
threshold_critical = mean + 3 * std

print("\nGaussian Thresholds:")
print("Healthy Limit :", threshold_healthy)
print("Critical Limit:", threshold_critical)

# ============================================================
# 1️⃣2️⃣ CURRENT MACHINE STATUS
# ============================================================

current_error = recon_errors[-1]

if current_error < threshold_healthy:
    current_status = "Healthy"
elif current_error < threshold_critical:
    current_status = "Degrading"
else:
    current_status = "Critical"

# ============================================================
# 1️⃣3️⃣ HEALTH INDEX (SMOOTHED)
# ============================================================

min_err = recon_errors.min()
max_err = recon_errors.max()

health_trend = 1 - ((recon_errors - min_err) / (max_err - min_err))
health_trend = np.clip(health_trend, 0, 1)

health_trend = pd.Series(health_trend).rolling(10).mean()
current_health_index = health_trend.iloc[-1]

# ============================================================
# 1️⃣4️⃣ OUTPUT
# ============================================================

print("\n==============================")
print("CURRENT MACHINE STATUS")
print("==============================")
print("Reconstruction Error :", round(current_error, 6))
print("Health Index (0-1)   :", round(float(current_health_index), 4))
print("Machine Status       :", current_status)
print("==============================\n")

# ============================================================
# 1️⃣5️⃣ PLOT HEALTH TREND
# ============================================================

plt.figure(figsize=(12,6))
plt.plot(health_trend)
plt.title("Smoothed Machine Health Index Over Time")
plt.xlabel("Time")
plt.ylabel("Health Index (0-1)")
plt.grid(True)
plt.show()
