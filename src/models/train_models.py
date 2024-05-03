import tensorflow
import dagshub
import mlflow
import os

import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.preprocessing import MinMaxScaler
from model_data import prepare_train_test_data
from definitions import ROOT_DIR
from dotenv import load_dotenv
from src.serve.stations import get_stations

load_dotenv()


def create_station_model(input_shape):
    model = tensorflow.keras.models.Sequential()

    model.add(tensorflow.keras.layers.GRU(32, return_sequences=True, input_shape=input_shape))

    model.add(tensorflow.keras.layers.GRU(32, activation='relu'))

    model.add(tensorflow.keras.layers.Dense(16, activation='relu'))

    model.add(tensorflow.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def save_scaler(client, scaler_type, scaler, station_name):
    metadata = {
        "station_name": station_name,
        "scaler_type": scaler_type,
        "expected_features": scaler.n_features_in_,
        "feature_range": scaler.feature_range,
    }

    scaler = mlflow.sklearn.log_model(
        sk_model=scaler,
        artifact_path=f"models/{station_name}/{scaler_type}",
        registered_model_name=f"{scaler_type}={station_name}",
        metadata=metadata,
    )

    scaler_version = client.create_model_version(
        name=f"{scaler_type}={station_name}",
        source=scaler.model_uri,
        run_id=scaler.run_id
    )

    client.transition_model_version_stage(
        name=f"{scaler_type}={station_name}",
        version=scaler_version.version,
        stage="staging",
    )


def main():
    station_directory = ROOT_DIR + '/data/processed/'

    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(os.getenv("DAGSHUB_REPO_NAME"), os.getenv("DAGSHUB_USERNAME"), mlflow=True)
    mlflow.set_tracking_uri(os.getenv("DAGSHUB_URI"))
    ml_flow_client = MlflowClient()

    station_names = get_stations()
    for station_name in station_names:
        station_name = station_name['name']
        mlflow.start_run(run_name=f"run={station_name}")
        mlflow.tensorflow.autolog()

        window_size = 24
        batch_size = 32
        epochs = 2

        abs_scaler = MinMaxScaler()
        other_scaler = MinMaxScaler()

        X_train, y_train, X_test, y_test = prepare_train_test_data(
            station_directory,
            station_name,
            window_size,
            abs_scaler,
            other_scaler,
        )

        input_shape = (X_train.shape[1], X_train.shape[2])

        # Create the model
        station_model = create_station_model(input_shape)

        # Train the model
        station_model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size)

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("train_dataset_size", len(X_train) + len(y_train))

        station_model = mlflow.sklearn.log_model(
            sk_model=station_model,
            artifact_path=f"models/{station_name}/model",
            registered_model_name=f"model={station_name}",
        )

        # Create model version
        model_version = ml_flow_client.create_model_version(
            name=f"model={station_name}",
            source=station_model.model_uri,
            run_id=station_model.run_id
        )

        # Create model version stage
        ml_flow_client.transition_model_version_stage(
            name=f"model={station_name}",
            version=model_version.version,
            stage="staging",
        )

        # Save scalers
        save_scaler(ml_flow_client, "abs_scaler", abs_scaler, station_name)
        save_scaler(ml_flow_client, "other_scaler", other_scaler, station_name)

        mlflow.end_run()


if __name__ == "__main__":
    main()
