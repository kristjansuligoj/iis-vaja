import dagshub
import mlflow
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from model_data import prepare_train_test_data
from mlflow import MlflowClient
from common.common import make_directory_if_missing
from src.models.mlflow_client import download_model, download_scaler
from definitions import ROOT_DIR
from dotenv import load_dotenv
from src.serve.stations import get_stations

load_dotenv()


def get_metrics(inverse_y_test, inverse_model_predictions):
    mae = mean_absolute_error(inverse_y_test, inverse_model_predictions)
    mse = mean_squared_error(inverse_y_test, inverse_model_predictions)
    evs = explained_variance_score(inverse_y_test, inverse_model_predictions)

    return mae, mse, evs


def get_model_metrics(model, scaler, predict_X, predict_y):
    # Predict with model
    predictions = model.predict(predict_X)

    # Inverse standardisation
    inverse_predictions = scaler.inverse_transform(predictions)
    inverse_y = scaler.inverse_transform(predict_y.reshape(-1, 1)).flatten()

    # Calculate metrics of TEST data
    return get_metrics(inverse_y, inverse_predictions)


def replace_production_model(station_name):
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

        # Read models and scalers from mlflow
        print(f"\nDownloading latest staging model {station_name}\n")

        stage = "staging"
        station_model = download_model(station_name, stage)
        abs_scaler = download_scaler(station_name, "abs_scaler", stage)
        other_scaler = download_scaler(station_name, "other_scaler", stage)

        print(f"\nDownloading production model {station_name}\n")

        stage = "production"
        station_model_production = download_model(station_name, stage)
        abs_scaler_production = download_scaler(station_name, "abs_scaler", stage)
        other_scaler_production = download_scaler(station_name, "other_scaler", stage)

        if station_model is None or abs_scaler is None or other_scaler is None:
            print(f"Model or scaler was not downloaded properly. Skipping {station_name}")
            mlflow.end_run()
            continue

        if station_model_production is None or abs_scaler_production is None or other_scaler_production is None:
            replace_production_model(station_name)
            print(f"Production model does not exist. Replacing with latest staging model. Skipping evaluation.")
            mlflow.end_run()
            continue

        X_train, y_train, X_test, y_test = prepare_train_test_data(
            station_directory,
            station_name,
            window_size,
            abs_scaler,
            other_scaler
        )

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("train_dataset_size", len(X_train) + len(y_train))

        print(f"\nTraining latest staging model {station_name}\n")

        # Get metrics for latest model
        mae, mse, evs = get_model_metrics(
            station_model,
            abs_scaler,
            X_test,
            y_test,
        )

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("evs", evs)

        print(f"\nTraining production model {station_name}\n")

        # Get metrics for model in production
        mae_production, mse_production, evs_production = get_model_metrics(
            station_model_production,
            abs_scaler_production,
            X_test,
            y_test,
        )

        # Check if new model is better than production, and replace it if it is
        if mae < mae_production:
            print("New model was better than the one in production. Replacing it now . . .")
            replace_production_model(station_name)

        # Save metrics to file
        print(f"\nSaving metrics to file for model {station_name}\n")
        make_directory_if_missing(os.path.join(ROOT_DIR, "reports"))
        with open(ROOT_DIR + '/reports/model=' + station_name + '-metrics.txt', 'w', encoding='utf-8') as f:
            f.write(
                f'Mean average error: {mae}\nMean square error: {mse}\nExplained variance score: {evs}\n')

        mlflow.end_run()


if __name__ == "__main__":
    main()
