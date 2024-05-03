import dagshub
import mlflow
import os
import joblib
from stations import get_stations
from dotenv import load_dotenv
from definitions import ROOT_DIR
from common.common import make_directory_if_missing
from src.models.mlflow_client import download_model, download_scaler

load_dotenv()


def is_model_already_downloaded(model_path, abs_scaler_path):
    return os.path.exists(model_path) and os.path.exists(abs_scaler_path)


def download_models():
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(os.getenv("DAGSHUB_REPO_NAME"), os.getenv("DAGSHUB_USERNAME"), mlflow=True)
    mlflow.set_tracking_uri(os.getenv("DAGSHUB_URI"))

    for station in get_stations():
        model_path = os.path.join(ROOT_DIR, "models", station['name'], "production_model.h5")
        abs_scaler_path = os.path.join(ROOT_DIR, "models", station['name'], "production_abs_scaler.gz")

        # Check if model is already downloaded, and skip this station if it is
        if is_model_already_downloaded(model_path, abs_scaler_path):
            continue

        try:
            # Download the models and scaler
            station_model = download_model(station['name'], "production")
            station_abs_scaler = download_scaler(station['name'], "abs_scaler", "production")

            # Create station model directory
            station_model_directory = os.path.join(ROOT_DIR, "models", station["name"])
            make_directory_if_missing(station_model_directory)

            # Save the models to the newly created directory
            joblib.dump(station_model, model_path)
            joblib.dump(station_abs_scaler, abs_scaler_path)
        except mlflow.exceptions.RestException:
            print(f"There was an error downloading {station['name']}")
            continue


