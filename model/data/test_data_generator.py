import numpy as np
import pandas as pd
import os

np.random.seed(123)

NUM_MACHINES = 6
HOURS = 300
DATES = pd.date_range(start="2025-01-01", periods=HOURS, freq="h")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

records = []

machine_states = {
    "M01": "healthy",
    "M02": "healthy",
    "M03": "degrading",
    "M04": "degrading",
    "M05": "critical",
    "M06": "failing"
}

cycle_length = 2285  # same as training

for machine_id, state in machine_states.items():

    base_vibration = np.random.uniform(0.2, 0.4)
    base_temp = np.random.uniform(55, 65)

    maint_timer = 0

    # 🔥 KEY FIX: force cycle position
    if state == "healthy":
        total_operating_hours = np.random.randint(0, 200)

    elif state == "degrading":
        total_operating_hours = np.random.randint(900, 1400)

    elif state == "critical":
        total_operating_hours = np.random.randint(1800, 2100)

    else:  # failing
        total_operating_hours = np.random.randint(2100, 2280)

    # 🔥 initial degradation still matters
    if state == "healthy":
        degradation = 0.02
    elif state == "degrading":
        degradation = 0.2
    elif state == "critical":
        degradation = 0.7
    else:
        degradation = 0.9

    for ts in DATES:
        total_operating_hours += 1
        maint_timer += 1

        # degradation speed
        if state == "healthy":
            wear_step = np.random.uniform(1e-4, 2e-4)
        elif state == "degrading":
            wear_step = np.random.uniform(2e-4, 4e-4)
        else:
            wear_step = np.random.uniform(5e-4, 9e-4)

        degradation += wear_step

        # sensor signals (keep yours but slightly stronger separation)
        vibration = base_vibration * (1 + (3 + 5 * degradation) * degradation) + np.random.normal(0, 0.02)
        process_temp = base_temp * (1 + (2 + 3 * degradation) * degradation) + np.random.normal(0, 1.0)

        machine_failure = 0
        maint_type = "None"

        # failure only for failing machine
        if state == "failing" and degradation > 1.1:
            machine_failure = 1
            maint_type = "corrective"
            degradation = 0.8
            maint_timer = 0

        torque = np.random.uniform(40, 70) * (1 + 0.2 * degradation)
        rpm = np.random.uniform(1200, 1800) * (1 - 0.25 * degradation)
        current = (torque * rpm) / 9000 + np.random.normal(0, 0.2)

        records.append([
            ts,
            machine_id,
            process_temp,
            np.random.uniform(20, 30),
            vibration,
            torque,
            rpm,
            current,
            total_operating_hours,
            maint_timer,
            maint_type,
            machine_failure,
            np.random.uniform(0, 0.2),
            current * 0.415
        ])

columns = [
    "timestamp", "machine_id", "process_temperature", "air_temperature",
    "vibration", "torque", "rpm", "current", "operating_hours",
    "time_since_last_maintenance", "last_maintenance_Type", "machine_failure",
    "idle_duration", "power_consumption"
]

df = pd.DataFrame(records, columns=columns)

save_path = os.path.join(DATA_DIR, "inference_test_data.csv")
df.to_csv(save_path, index=False)

print("Saved:", save_path)
print("Failures:", df["machine_failure"].sum())