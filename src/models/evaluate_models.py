import dagshub
import mlflow
import os
import onnxruntime
import onnx
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from model_data import prepare_train_test_data
from mlflow import MlflowClient
from common.common import make_directory_if_missing
from src.models.mlflow_client import download_model, download_scaler, get_latest_model
from definitions import ROOT_DIR
from dotenv import load_dotenv
from src.serve.stations import get_stations

load_dotenv()


def get_metrics(inverse_y_test, inverse_model_predictions):
    mae = mean_absolute_error(inverse_y_test, inverse_model_predictions)
    mse = mean_squared_error(inverse_y_test, inverse_model_predictions)
    evs = explained_variance_score(inverse_y_test, inverse_model_predictions)

    return mae, mse, evs


def get_model_metrics(y_predictions, y_true, scaler):
    inverse_predictions = scaler.inverse_transform(y_predictions)

    mse = mean_squared_error(y_true, inverse_predictions)
    mae = mean_absolute_error(y_true, inverse_predictions)
    evs = explained_variance_score(y_true, inverse_predictions)

    return mse, mae, evs


def replace_prod_model(station_name):
    model_name = f"model={station_name}"
    abs_scaler_name = f"abs_scaler={station_name}"
    other_scaler_name = f"other_scaler={station_name}"

    try:
        client = MlflowClient()

        # Get model and scaler latest staging version
        model_version = client.get_latest_versions(name=model_name, stages=["staging"])[0].version
        abs_scaler_version = client.get_latest_versions(name=abs_scaler_name, stages=["staging"])[0].version
        other_scaler_version = client.get_latest_versions(name=other_scaler_name, stages=["staging"])[0].version

        # Update production model and scaler
        client.transition_model_version_stage(model_name, model_version, "production")
        client.transition_model_version_stage(abs_scaler_name, abs_scaler_version, "production")
        client.transition_model_version_stage(other_scaler_name, other_scaler_version, "production")
    except IndexError:
        print(f"There was an error replacing production model {model_name}")
        return None


def main():
    station_directory = ROOT_DIR + '/data/processed/'

    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(os.getenv("DAGSHUB_REPO_NAME"), os.getenv("DAGSHUB_USERNAME"), mlflow=True)
    mlflow.set_tracking_uri(os.getenv("DAGSHUB_URI"))

    station_names = get_stations()
    for station_name in station_names:
        station_name = station_name['name']
        print(f"\nEvaluating model {station_name}\n")
        mlflow.start_run(run_name=f"run={station_name}")
        mlflow.tensorflow.autolog()

        window_size = 24
        epochs = 10

        print(f"\nDownloading latest staging model {station_name}\n")

        staging_model_path, staging_abs_scaler, staging_other_scaler = get_latest_model(station_name, "staging")

        if staging_model_path is None or staging_abs_scaler is None or staging_other_scaler is None:
            print(f"Model or scaler was not downloaded properly. Skipping {station_name}")
            mlflow.end_run()
            continue

        staging_model = onnxruntime.InferenceSession(staging_model_path)

        print(f"\nDownloading production model {station_name}\n")

        prod_model_path, prod_abs_scaler, prod_other_scaler = get_latest_model(station_name, "production")

        if prod_model_path is None or prod_abs_scaler is None or prod_other_scaler is None:
            replace_prod_model(station_name)
            print(f"Production model does not exist. Replacing with latest staging model. Skipping evaluation.")
            mlflow.end_run()
            continue

        prod_model = onnxruntime.InferenceSession(prod_model_path)

        print(f"\nCreating train and test data.\n")

        X_train, y_train, X_test, y_test = prepare_train_test_data(
            station_directory,
            station_name,
            window_size,
            staging_abs_scaler,
            staging_other_scaler
        )

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("train_dataset_size", len(X_train) + len(y_train))

        print(f"\nTraining latest staging model {station_name}\n")

        staging_model_predictions = staging_model.run(
            ["output"],
            {"input": X_test}
        )[0]

        # Get metrics for latest model
        mae_staging, mse_staging, evs_staging = get_model_metrics(
            staging_model_predictions,
            y_test,
            staging_abs_scaler,
        )

        mlflow.log_metric("mae", mae_staging)
        mlflow.log_metric("mse", mse_staging)
        mlflow.log_metric("evs", evs_staging)

        print(f"\nTraining production model {station_name}\n")

        X_train, y_train, X_test, y_test = prepare_train_test_data(
            station_directory,
            station_name,
            window_size,
            prod_abs_scaler,
            prod_other_scaler
        )

        prod_model_predictions = prod_model.run(
            ["output"],
            {"input": X_test}
        )[0]

        # Get metrics for latest model
        mae_prod, mse_prod, evs_prod = get_model_metrics(
            prod_model_predictions,
            y_test,
            prod_abs_scaler,
        )

        # Check if new model is better than production, and replace it if it is
        if mae_staging < mae_prod:
            print("New model was better than the one in production. Replacing it now . . .")
            replace_prod_model(station_name)

        # Save metrics to file
        print(f"\nSaving metrics to file for model {station_name}\n")

        make_directory_if_missing(os.path.join(ROOT_DIR, "reports"))
        with open(ROOT_DIR + '/reports/model=' + station_name + '-metrics.txt', 'w', encoding='utf-8') as f:
            f.write(
                f'Mean average error: {mae_staging}\n'
                f'Mean square error: {mse_staging}\n'
                f'Explained variance score: {evs_staging}\n'
            )

        mlflow.end_run()


if __name__ == "__main__":
    main()
