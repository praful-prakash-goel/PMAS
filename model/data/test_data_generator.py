import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from backend.models import Machine
from backend.database import SessionLocal

# Configuration
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "inference_test_data.csv")
STATE_PATH = os.path.join(DATA_DIR, "machine_state.json")

# Probability Factors (tuned for 5-second intervals)
P_HEALTHY_TO_DEGRADING = 0.01  # 1% chance (~10 mins expected time)
P_DEGRADING_TO_CRITICAL = 0.04 # 4% chance (~1.5 mins expected time)

def get_machine_ids():
    db = SessionLocal()
    try:
        machines = db.query(Machine).all()
        return [m.machine_id for m in machines]
    finally:
        db.close()
    
def generate_data(verbose: int = 0):
    machine_ids = get_machine_ids()
    
    # Initialize or Load State
    if not os.path.exists(STATE_PATH):
        machine_state = {}

        for i, machine_id in enumerate(machine_ids):
            if i == 0:
                state = "critical"
                degradation = 0.7
                op_hours = np.random.randint(1200, 1600)

            elif i == 1:
                state = "degrading"
                degradation = 0.3
                op_hours = np.random.randint(400, 800)

            else:
                state = "healthy"
                degradation = np.random.uniform(0.02, 0.05)
                op_hours = np.random.randint(0, 200)

            machine_state[machine_id] = {
                "state": state,
                "degradation": degradation,
                "base_vib": np.random.uniform(0.2, 0.4),
                "base_temp": np.random.uniform(55, 65),
                "op_hours": op_hours,
                "maint_timer": 0
            }
    else:
        with open(STATE_PATH, 'r') as f:
            machine_state = json.load(f)

    # Add new machines
    for machine_id in machine_ids:
        if machine_id not in machine_state:
            machine_state[machine_id] = {
                "state": "healthy",
                "degradation": np.random.uniform(0.02, 0.05),
                "base_vib": np.random.uniform(0.2, 0.4),
                "base_temp": np.random.uniform(55, 65),
                "op_hours": np.random.randint(0, 100),
                "maint_timer": 0
            }

    # Remove deleted machines
    for machine_id in list(machine_state.keys()):
        if machine_id not in machine_ids:
            del machine_state[machine_id]

    current_timestamp = pd.Timestamp.now().round('s')
    records = []

    for machine_id, data in machine_state.items():
        state = data["state"]
        degradation = data["degradation"]
        
        # 1. Evaluate State Transitions based on Probability
        if state == "healthy" and np.random.rand() < P_HEALTHY_TO_DEGRADING:
            state = "degrading"
            degradation = max(degradation, 0.2)
        elif state == "degrading" and np.random.rand() < P_DEGRADING_TO_CRITICAL:
            state = "critical"
            degradation = max(degradation, 0.7)

        # 2. Apply continuous wear and tear (Cap at 1.25 to prevent extreme math anomalies)
        if state == "healthy":
            wear_step = np.random.uniform(1e-4, 5e-4)
        elif state == "degrading":
            wear_step = np.random.uniform(1e-3, 3e-3)
        else: # critical
            # Accelerated wear so a critical machine hits failure (1.1) within a reasonable timeframe
            wear_step = np.random.uniform(1e-2, 3e-2) 

        if degradation < 1.25:
            degradation += wear_step
            
        data["op_hours"] += (5 / 3600)
        data["maint_timer"] += (5 / 3600)

        # 3. Generate Sensor Signals
        vibration = data["base_vib"] * (1 + (3 + 5 * degradation) * degradation) + np.random.normal(0, 0.02)
        process_temp = data["base_temp"] * (1 + (2 + 3 * degradation) * degradation) + np.random.normal(0, 1.0)
        
        machine_failure = 0
        maint_type = "None"

        # 4. Check for catastrophic failure trigger (Requires Critical State AND High Degradation)
        if state == "critical" and degradation > 1.1:
            machine_failure = 1
            maint_type = "corrective"

        torque = np.random.uniform(40, 70) * (1 + 0.2 * degradation)
        rpm = np.random.uniform(1200, 1800) * (1 - 0.25 * degradation)
        current = (torque * rpm) / 9000 + np.random.normal(0, 0.2)

        # Append to current run records
        records.append([
            current_timestamp, machine_id, process_temp, np.random.uniform(20, 30),
            vibration, torque, rpm, current, data["op_hours"],
            data["maint_timer"], maint_type, machine_failure,
            np.random.uniform(0, 0.2), current * 0.415
        ])

        # Save updated variables back to state dictionary
        data["state"] = state
        data["degradation"] = degradation

    # Save updated state to JSON for the next run
    with open(STATE_PATH, 'w') as f:
        json.dump(machine_state, f, indent=4)

    # Create DataFrame
    columns = [
        "timestamp", "machine_id", "process_temperature", "air_temperature",
        "vibration", "torque", "rpm", "current", "operating_hours",
        "time_since_last_maintenance", "last_maintenance_Type", "machine_failure",
        "idle_duration", "power_consumption"
    ]
    df = pd.DataFrame(records, columns=columns)

    # Append to CSV
    file_exists = os.path.exists(CSV_PATH)
    temp_path = CSV_PATH + ".tmp"
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, CSV_PATH)

    if verbose == 1:
        print(f"[{current_timestamp}] Data appended to {CSV_PATH}. Failures in this batch: {df['machine_failure'].sum()}")
        
if __name__ == '__main__':
    generate_data(verbose=1)