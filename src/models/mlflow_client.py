import mlflow


def download_model(station_name, stage):
    model_name = f"model={station_name}"

    try:
        client = mlflow.MlflowClient()

        # Get model latest staging source
        latest_model_version_source = client.get_latest_versions(name=model_name, stages=[stage])[0].source

        # Load the model by its source
        return mlflow.sklearn.load_model(latest_model_version_source)
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
