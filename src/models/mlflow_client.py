import mlflow
from common.common import make_directory_if_missing
from definitions import ROOT_DIR
import os
import joblib
import onnx


def download_model(station_name, stage):
    model_name = f"model={station_name}"

    try:
        client = mlflow.MlflowClient()

        # Get model latest staging source
        latest_model_version_source = client.get_latest_versions(name=model_name, stages=[stage])[0].source

        # Load the model by its source
        return mlflow.onnx.load_model(latest_model_version_source)
    except IndexError:
        print(f"There was an error downloading {model_name} in {stage}")
        return None


def download_scaler(station_name, scaler_type, stage):
    scaler_name = f"{scaler_type}={station_name}"

    try:
        client = mlflow.MlflowClient()

        # Get scaler latest staging source
        latest_scaler_source = client.get_latest_versions(name=scaler_name, stages=[stage])[0].source

        # Load the scaler by its source
        return mlflow.sklearn.load_model(latest_scaler_source)
    except IndexError:
        print(f"There was an error downloading {scaler_name} in {stage}")
        return None


def get_latest_model(station_name, stage):
    # Download model and scalers for station
    model = download_model(station_name, stage)
    other_scaler = download_scaler(station_name, "other_scaler", stage)
    abs_scaler = download_scaler(station_name, "abs_scaler", stage)

    # Create model directory if it does not exist
    base_station_directory = os.path.join(ROOT_DIR, "models", station_name)
    make_directory_if_missing(base_station_directory)

    # Save other_scaler
    other_scaler_path = os.path.join(base_station_directory, f"other_scaler={station_name}.gz")
    joblib.dump(other_scaler, other_scaler_path)

    # Save abs_scaler
    abs_scaler_path = os.path.join(base_station_directory, f"abs_scaler={station_name}.gz")
    joblib.dump(abs_scaler, abs_scaler_path)

    # Save model
    model_path = os.path.join(base_station_directory, f"model={station_name}.onnx")
    onnx.save_model(model, model_path)

    return model_path, other_scaler, abs_scaler
