import dagshub
import mlflow
import os

from stations import get_stations
from dotenv import load_dotenv
from definitions import ROOT_DIR
from src.models.mlflow_client import get_latest_model

load_dotenv()


def is_model_already_downloaded(model_path, abs_scaler_path):
    return os.path.exists(model_path) and os.path.exists(abs_scaler_path)


def download_models():
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(os.getenv("DAGSHUB_REPO_NAME"), os.getenv("DAGSHUB_USERNAME"), mlflow=True)
    mlflow.set_tracking_uri(os.getenv("DAGSHUB_URI"))

    print("Checking if models exist locally, and download them otherwise.")

    for station in get_stations():
        model_path = os.path.join(ROOT_DIR, "models", station['name'], f"model={station['name']}.onnx")
        abs_scaler_path = os.path.join(ROOT_DIR, "models", station['name'], f"abs_scaler={station['name']}.gz")

        # Check if model is already downloaded, and skip this station if it is
        if is_model_already_downloaded(model_path, abs_scaler_path):
            print(f"Model for {station['name']} already loaded. Skipping. . .")
            continue

        try:
            get_latest_model(station['name'], "production")
        except mlflow.exceptions.RestException:
            print(f"There was an error downloading {station['name']}")
            continue

