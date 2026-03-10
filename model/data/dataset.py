import numpy as np
import pandas as pd
import os

np.random.seed(42)

NUM_MACHINES = 5
YEARS = 10
FREQ = "h"

DATES = pd.date_range(
    start="2015-01-01",
    end="2024-12-31 23:00:00",
    freq=FREQ
)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

records = []

for machine_id in range(1, NUM_MACHINES + 1):

    # Machine-specific baselines
    base_temp = np.random.uniform(55, 65)
    base_vibration = np.random.uniform(0.2, 0.4)
    base_rpm = np.random.uniform(1200, 1800)
    base_torque = np.random.uniform(40, 70)

    operating_hours = 0
    hours_since_maintenance = 0
    degradation = 0
    maintenance_type = "None"

    for i, ts in enumerate(DATES):

        operating_hours += 1
        hours_since_maintenance += 1

        # Slow degradation (hourly)
        degradation += np.random.uniform(1e-6, 5e-6)

        # Maintenance decision once per day
        if ts.hour == 0 and hours_since_maintenance > np.random.randint(3000, 4500):
            hours_since_maintenance = 0
            degradation *= np.random.uniform(0.3, 0.6)
            maintenance_type = np.random.choice(
                ["Preventive", "Corrective"], p=[0.75, 0.25]
            )
        else:
            maintenance_type = "None"

        # Environment
        air_temp = np.random.uniform(18, 35)

        # Operational behavior
        load_factor = np.random.uniform(0.6, 1.0)

        rpm = base_rpm * load_factor * (1 - 0.15 * degradation) + np.random.normal(0, 15)
        torque = base_torque * load_factor * (1 + 0.25 * degradation) + np.random.normal(0, 2)

        vibration = base_vibration * (1 + 3 * degradation) + np.random.normal(0, 0.01)
        process_temp = base_temp * (1 + 1.5 * degradation) + np.random.normal(0, 1.0)

        # Idle behavior (≤ 1 hour)
        idle_duration = max(0, min(1, np.random.normal(0.1 + 0.6 * degradation, 0.1)))

        # Electrical
        current = (torque * rpm) / 9000 + np.random.normal(0, 0.3)
        power_consumption = current * 415 / 1000  # kW

        records.append([
            ts,
            f"M{machine_id:02d}",
            process_temp,
            air_temp,
            vibration,
            torque,
            rpm,
            current,
            operating_hours,
            hours_since_maintenance,
            maintenance_type,
            idle_duration,
            power_consumption
        ])

columns = [
    "timestamp",
    "machine_id",
    "process_temperature",
    "air_temperature",
    "vibration",
    "torque",
    "rpm",
    "current",
    "operating_hours",
    "time_since_last_maintenance",
    "last_maintenance_Type",
    "idle_duration",
    "power_consumption"
]

df = pd.DataFrame(records, columns=columns)

df.to_csv(f"{DATA_DIR}/Predictive_Maintenance_Synthetic_Data.csv", index=False)

print(df.head())
print("Total rows:", len(df))
