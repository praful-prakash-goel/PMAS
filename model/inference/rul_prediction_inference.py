import os
import pandas as pd
import numpy as np
import argparse
from ..rul_model import MODEL_DIR, load_data, engineer_features, get_feature_cols
import joblib
import xgboost as xgb

DATA_PATH = os.path.join(MODEL_DIR, "data", "inference_test_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "saved_models", "rul_final_model.json")
CONFIG_PATH = os.path.join(MODEL_DIR ,"saved_models", "model_config.pkl")
OUTPUT_PATH = os.path.join(MODEL_DIR, "results", "inference_results.csv")
FEATURE_PATH = os.path.join(MODEL_DIR, "saved_models", "feature_cols.pkl")

def predict(model_config, data_path: str = DATA_PATH, model_path: str = MODEL_PATH, feature_path: str = FEATURE_PATH):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    df = load_data(data_path, verbose=0)
    df_engineered = engineer_features(df, config=model_config, verbose=0)
    feature_cols = joblib.load(feature_path)
    
    X = df_engineered[feature_cols].values
    preds_log = model.predict(X)

    preds = np.expm1(preds_log)  # convert back to hours

    df_engineered["predicted_RUL"] = np.clip(
        preds, 0, model_config["feature_engineering"]["rul_cap"]
    )
    
    for col in feature_cols:
        if col not in df_engineered.columns:
            df_engineered[col] = 0
        
    latest_status = (
        df_engineered.sort_values(["machine_id", "timestamp"])
        .groupby("machine_id")
        .tail(1)
        .copy()
    )
    
    return latest_status

def map_health_status(rul_hours):
    if rul_hours <= 100:
        return "CRITICAL"
    elif rul_hours <= 350:
        return "DEGRADING"
    else:
        return "HEALTHY"

def save_inference_report(df_with_preds: pd.DataFrame, output_path: str = OUTPUT_PATH, verbose: int = 0):
    """
    Takes the model output (DataFrame with 'predicted_RUL') 
    and saves the final maintenance report.
    """
    # We keep the core sensor context and the prediction
    report_df = df_with_preds[[
        "machine_id", "timestamp", "operating_hours", 
        "vibration_zscore", "predicted_RUL"
    ]].copy()

    # rename for Clarity
    report_df = report_df.rename(columns={"predicted_RUL": "RUL_predicted_hours"})

    # Apply Transformations
    report_df["RUL_predicted_hours"] = report_df["RUL_predicted_hours"].round(2)
    report_df["RUL_predicted_days"]  = (report_df["RUL_predicted_hours"] / 24).round(2)
    
    # Next maintenance is physically the same as RUL
    report_df["next_maintenance_days"] = report_df["RUL_predicted_days"]
    
    # Map Health Status
    # Using .apply for smaller fleet or np.select for larger performance
    report_df["health_status"] = report_df["RUL_predicted_hours"].apply(map_health_status)

    # Add Expected Failure Date
    # Converts the RUL hours into an actual calendar date/time
    report_df["expected_failure_date"] = (
        pd.to_datetime(report_df["timestamp"]) + 
        pd.to_timedelta(report_df["RUL_predicted_hours"], unit='h')
    ).dt.strftime('%Y-%m-%d %H:%M')

    # Save and Return
    report_df.to_csv(output_path, index=False)
    if verbose == 1:
        print(f"\n[SUCCESS] Inference report saved to: {output_path}")
    
    return report_df
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RUL prediction with LOMO-CV and XGBoost")
    parser.add_argument("--data",   default=DATA_PATH, help="Path to CSV file")
    
    args = parser.parse_args()
    
    model_config = joblib.load(CONFIG_PATH)
    summary_df = predict(model_config=model_config, data_path=DATA_PATH, model_path=MODEL_PATH, feature_path=FEATURE_PATH)
    save_inference_report(summary_df, output_path=OUTPUT_PATH, verbose=1)
    