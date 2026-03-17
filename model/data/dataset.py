import numpy as np
import pandas as pd
import os

np.random.seed(42)

NUM_MACHINES = 8
HOURS = 24 * 365 * 2   # 2 years
DATES = pd.date_range(start="2024-01-01", periods=HOURS, freq="h")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

records = []

for machine_id in range(1, NUM_MACHINES + 1):

    base_vibration = np.random.uniform(0.2, 0.4)
    base_temp = np.random.uniform(55, 65)

    total_operating_hours = 0
    maint_timer = 0
    
    # 🔥 Start from random health state
    degradation = np.random.uniform(0.0, 0.3)

    for ts in DATES:
        total_operating_hours += 1
        maint_timer += 1

        # ─────────────────────────────
        # SMOOTH DEGRADATION
        # ─────────────────────────────
        wear_step = 3e-4 + np.random.normal(0, 7e-5)
        wear_step = max(wear_step, 1e-5)
        degradation += wear_step

        # ─────────────────────────────
        # CRITICAL ZONE AMPLIFICATION
        # ─────────────────────────────
        critical_factor = 1.0
        if degradation > 0.5:
            critical_factor = 2.0

        # ─────────────────────────────
        # SENSOR GENERATION
        # ─────────────────────────────
        vibration = base_vibration * (1 + 5 * degradation * critical_factor) + np.random.normal(0, 0.015)
        process_temp = base_temp * (1 + 2.5 * degradation * critical_factor) + np.random.normal(0, 0.8)

        torque = np.random.uniform(40, 70) * (1 + 0.12 * degradation)
        rpm = np.random.uniform(1200, 1800) * (1 - 0.18 * degradation)
        current = (torque * rpm) / 9000 + np.random.normal(0, 0.15)

        machine_failure = 0
        maint_type = "None"

        # ─────────────────────────────
        # FAILURE EVENT
        # ─────────────────────────────
        if vibration > base_vibration * 3.2:
            machine_failure = 1
            maint_type = "corrective"

            # 🔥 Partial reset (NOT zero)
            degradation = np.random.uniform(0.2, 0.4)
            maint_timer = 0

        # ─────────────────────────────
        # RARE PREVENTIVE MAINTENANCE
        # ─────────────────────────────
        elif maint_timer > np.random.randint(7000, 10000):
            maint_type = "preventive"
            degradation *= 0.6
            maint_timer = 0

        records.append([
            ts,
            f"M{machine_id:02d}",
            process_temp,
            np.random.uniform(20, 30),  # air temp
            vibration,
            torque,
            rpm,
            current,
            total_operating_hours,
            maint_timer,
            maint_type,
            machine_failure,
            np.random.uniform(0, 0.2),  # idle duration
            current * 0.415
        ])

columns = [
    "timestamp", "machine_id", "process_temperature", "air_temperature",
    "vibration", "torque", "rpm", "current", "operating_hours",
    "time_since_last_maintenance", "last_maintenance_Type", "machine_failure",
    "idle_duration", "power_consumption"
]

df = pd.DataFrame(records, columns=columns)

save_path = os.path.join(DATA_DIR, "Predictive_Maintenance_Synthetic_Data.csv")
df.to_csv(save_path, index=False)

print(f"Dataset saved at: {save_path}")
print(f"Rows: {len(df)}")
print(f"Failures: {df['machine_failure'].sum()}")