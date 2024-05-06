from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from definitions import ROOT_DIR
from stations import get_station_data
from download_models import download_models

from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import os
import onnxruntime
from src.database.connector import insert_data


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


def prepare_data(df, other_scaler, abs_scaler):
    columns_of_interest = [
        'available_bike_stands',
        'temperature',
        'relative_humidity',
        'dew_point',
        'apparent_temperature',
        'precipitation_probability',
        'rain',
        'surface_pressure',
        'bike_stands'
    ]

    df = df[columns_of_interest].values

    test_data = df

    abs_test_data = test_data[:, 0]

    other_test_data = test_data[:, 1:]

    # Normalization
    normalized_other_test_data = other_scaler.transform(other_test_data)

    normalized_abs_test_data = abs_scaler.transform(abs_test_data.reshape(-1, 1))

    X_predict = np.column_stack([
        normalized_other_test_data,
        normalized_abs_test_data
    ])

    return X_predict.reshape(1, X_predict.shape[1], X_predict.shape[0])


def main():
    app = Flask(__name__)
    CORS(
        app,
        resources={
            r"/*": {
                "origins": "https://p01--iis-client--462724rjs8tc.code.run",
                "supports_credentials": True,
                "Access-Control-Allow-Credentials": True,
            }
        }
    )

    download_models()

    @app.route('/api/predict/<int:station_id>', methods=['GET'])
    def predict(station_id):
        station = get_station_data(station_id)

        print(f"Predicting for station {station['name']}. . .")

        model_path = os.path.join(ROOT_DIR, "models", station['name'], f"model={station['name']}.onnx")
        other_scaler_path = os.path.join(ROOT_DIR, "models", station['name'], f"other_scaler={station['name']}.gz")
        abs_scaler_path = os.path.join(ROOT_DIR, "models", station['name'], f"abs_scaler={station['name']}.gz")
        df_path = os.path.join(ROOT_DIR, "data", "processed", f"processed_data={station['name']}.csv")

        other_scaler = joblib.load(other_scaler_path)
        abs_scaler = joblib.load(abs_scaler_path)

        # Checks if weather data for this day does not exist yet
        if os.path.isfile(df_path):
            df = pd.read_csv(df_path)
            df['date'] = pd.to_datetime(df['date'])

            df['bike_stands'] = 22  # I NEED A DICTIONARY OF ALL STATIONS, SO THIS DATA IS AVAILABLE

            # Only save 19 latest data points, so we can get predictions for the next 7 hours (12 is window size)
            df = df.sort_values('date').head(32)

            predictions = []
            window_size = 24

            for i in range(7):
                start_index = i
                end_index = i + window_size
                df_window = df.iloc[start_index:end_index]

                X_predict = prepare_data(df_window, other_scaler, abs_scaler)

                model = onnxruntime.InferenceSession(model_path)

                model_predictions = model.run(
                    ["output"],
                    {"input": X_predict}
                )[0]

                inverse_model_predictions = abs_scaler.inverse_transform(model_predictions)

                predictions.append(float(inverse_model_predictions[0][0]))

            print("Predictions created. Sending data to database.")

            prediction = {
                'station': station['name'],
                'predictions': predictions,
                'date': datetime.now(),
            }

            insert_data('predictions', prediction)

            return jsonify({
                "station_name": station['name'],
                "station_bike_stands": station['bike_stands'],
                "predictions": predictions
            })

        return jsonify({"Message": "Data not available."})

    app.run(host='0.0.0.0', port=8080)


if __name__ == "__main__":
    main()
