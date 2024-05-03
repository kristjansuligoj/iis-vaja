from flask import Flask, jsonify
from flask_cors import CORS
from definitions import ROOT_DIR
from src.models.mlflow_client import download_model, download_scaler
from stations import get_station_data
from download_models import download_models

import pandas as pd
import numpy as np
import joblib
import os


expected_structure = [
    "date",
    "available_bike_stands"
]


def separate_datetime_columns(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S%z')
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['second'] = df['date'].dt.second
    return df.drop('date', axis=1)


def transform_columns(df, columns, scaler):
    for column in columns:
        df[column] = scaler.fit_transform(df[[column]])
    return df


def main():
    app = Flask(__name__)
    CORS(app)

    download_models()

    @app.route('/api/predict/<int:station_id>', methods=['GET'])
    def predict(station_id):
        station = get_station_data(station_id)

        model_path = os.path.join(ROOT_DIR, "models", station['name'], "production_model.h5")
        abs_scaler_path = os.path.join(ROOT_DIR, "models", station['name'], "production_abs_scaler.gz")

        model = joblib.load(model_path)
        abs_scaler = joblib.load(abs_scaler_path)

        weather_file_path = os.path.join(ROOT_DIR, "data", "processed", f"processed_data={station['name']}.csv")

        # Checks if weather data for this day does not exist yet
        if os.path.isfile(weather_file_path):
            df = pd.read_csv(weather_file_path)
            df['date'] = pd.to_datetime(df['date'])

            df['bike_stands'] = 22  # I NEED A DICTIONARY OF ALL STATIONS, SO THIS DATA IS AVAILABLE

            # Only save 19 latest data points, so we can get predictions for the next 7 hours (12 is window size)
            df = df.sort_values('date').head(32)

            columns_of_interest = [
                'temperature',
                'relative_humidity',
                'dew_point',
                'apparent_temperature',
                'precipitation_probability',
                'rain',
                'surface_pressure',
                'bike_stands'
            ]

            df = df[columns_of_interest]

            predictions = []
            window_size = 24

            for i in range(7):
                start_index = i
                end_index = i + window_size
                df_window = df.iloc[start_index:end_index]

                weather_data = df_window[columns_of_interest].values
                weather_data_reshaped = np.reshape(weather_data, (1, weather_data.shape[1], weather_data.shape[0]))
                prediction = model.predict(weather_data_reshaped)
                inverse_prediction = abs_scaler.inverse_transform(np.array(prediction).reshape(-1, 1))

                predictions.append(float(inverse_prediction[0][0]))

            return jsonify({
                "station_name": station['name'],
                "station_bike_stands": station['bike_stands'],
                "predictions": predictions
            })

        return jsonify({"Message": "Data not available."})

    app.run(host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main()
