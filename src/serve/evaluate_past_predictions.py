import dagshub
import mlflow
import os
import pandas as pd

from definitions import ROOT_DIR
from dotenv import load_dotenv
from src.serve.stations import get_stations
from src.database.connector import get_predictions_today_by_station
from datetime import timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

load_dotenv()


def assign_hour_to_predictions(predictions):
    predictions_with_hours = []
    for prediction_entry in predictions:
        prediction_datetime = prediction_entry['date']

        # Iterate over the predictions and assign an hour to each prediction
        assigned_hours = []
        for i, prediction in enumerate(prediction_entry['predictions']):
            date = prediction_datetime + timedelta(hours=i)
            assigned_hours.append({
                'date': date.strftime("%Y-%m-%d %H:%M:%S"),
                'pred': prediction,
            })

        predictions_with_hours.append(assigned_hours)

    return predictions_with_hours


def assign_true_value_to_predictions(predictions_with_hours, station_df):
    station_df['date'] = pd.to_datetime(station_df['date'])

    # Station can have rows that have duplicate date columns,
    # because data might not have been updated between API calls, so we remove them
    # https://stackoverflow.com/questions/43698764/removing-rows-with-a-duplicate-column-pandas-dataframe-python
    station_df.drop_duplicates('date', inplace=True)
    station_df.reset_index(inplace=True)

    # Sets the 'date' column as the index, so get_indexer looks at the date column
    # https://stackoverflow.com/questions/76739435/typeerror-cannot-compare-dtypes-int64-and-datetime64ns
    station_df = station_df.set_index(['date'])

    mapped_predictions = []
    for prediction_with_hours in predictions_with_hours:
        mapped_prediction = []
        for prediction in prediction_with_hours:
            prediction['date'] = pd.to_datetime(prediction['date'])

            # Find the nearest timestamp in 'station_actual_values' dataframe
            nearest_timestamp_index = station_df.index.get_indexer(
                [prediction['date']],
                method='nearest'
            )[0]

            # Extract the actual value from the nearest timestamp row
            row_with_nearest_timestamp = station_df.iloc[nearest_timestamp_index].to_dict()

            # Append the actual value to the mapped prediction
            prediction['true'] = row_with_nearest_timestamp['available_bike_stands']

            # Append the mapped prediction to the list
            mapped_prediction.append(prediction)

        # Append the mapped prediction to the mapped predictions list
        mapped_predictions.append(mapped_prediction)

    return mapped_predictions


def main():
    station_directory = ROOT_DIR + '/data/processed/'

    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(os.getenv("DAGSHUB_REPO_NAME"), os.getenv("DAGSHUB_USERNAME"), mlflow=True)
    mlflow.set_tracking_uri(os.getenv("DAGSHUB_URI"))

    station_names = get_stations()
    for station_name in station_names:
        station_name = station_name['name']

        # Get all predictions from the database for this station and day
        predictions_of_today = get_predictions_today_by_station(station_name)

        if not predictions_of_today:
            continue

        mlflow.start_run(run_name=f"experiment={station_name}")
        mlflow.tensorflow.autolog()

        # Read station data
        station_actual_values_path = os.path.join(station_directory, f"processed_data={station_name}.csv")
        station_df = pd.read_csv(station_actual_values_path)

        # Assign hours to predictions
        predictions_with_hours = assign_hour_to_predictions(predictions_of_today)

        # Assign true values to predictions
        predictions_dictionary = assign_true_value_to_predictions(predictions_with_hours, station_df)

        y_true = [entry['true'] for prediction_entry in predictions_dictionary for entry in prediction_entry]
        y_pred = [entry['pred'] for prediction_entry in predictions_dictionary for entry in prediction_entry]

        # Calculate the average error of model
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)

        # Save the result into mlflow as an experiment
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("evs", evs)

        mlflow.end_run()


if __name__ == "__main__":
    main()
