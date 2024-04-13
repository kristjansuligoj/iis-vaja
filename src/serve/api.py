from flask import Flask, jsonify, request
from flask_cors import CORS
from definitions import ROOT_DIR
from datetime import datetime, timedelta, time

import tensorflow
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


def get_station_data(station_id):
    stations = [
        {'id': 0, 'name': 'LIDL - TITOVA C.', 'bike_stands': 22},
        {'id': 1, 'name': 'NICEHASH - C. PROLETARSKIH BRIGAD', 'bike_stands': 22},
        {'id': 2, 'name': 'MLINSKA UL . - AVTOBUSNA POSTAJA', 'bike_stands': 22},
        {'id': 3, 'name': 'PARTIZANSKA C. - CANKARJEVA UL.', 'bike_stands': 22},
        {'id': 4, 'name': 'GORKEGA UL. - OŠ FRANCETA PREŠERNA', 'bike_stands': 22},
        {'id': 5, 'name': 'NKBM - TRG LEONA ŠTUKLJA', 'bike_stands': 22},
        {'id': 6, 'name': 'GOSPOSVETSKA C. - III. GIMNAZIJA', 'bike_stands': 22},
        {'id': 7, 'name': 'LJUBLJANSKA UL. - II. GIMNAZIJA', 'bike_stands': 22},
        {'id': 8, 'name': 'POŠTA - SLOMŠKOV TRG', 'bike_stands': 22},
        {'id': 9, 'name': 'NA POLJANAH - HEROJA ŠERCERJA', 'bike_stands': 22},
        {'id': 10, 'name': 'UM FGPA - LENT - SODNI STOLP', 'bike_stands': 22},
        {'id': 11, 'name': 'LIDL - KOROŠKA C.', 'bike_stands': 22},
        {'id': 12, 'name': 'PARTIZANSKA C. - ŽELEZNIŠKA POSTAJA', 'bike_stands': 22},
        {'id': 13, 'name': 'GOSPOSVETSKA C. - VRBANSKA C.', 'bike_stands': 22},
        {'id': 14, 'name': 'KOROŠKA C. - KOROŠKI VETER', 'bike_stands': 22},
        {'id': 15, 'name': 'JHMB – DVOETAŽNI MOST', 'bike_stands': 22},
        {'id': 16, 'name': 'RAZLAGOVA UL. - OBČINA', 'bike_stands': 22},
        {'id': 17, 'name': 'STROSSMAYERJEVA UL. - TRŽNICA', 'bike_stands': 22},
        {'id': 18, 'name': 'SPAR - TRŽNICA TABOR', 'bike_stands': 22},
        {'id': 19, 'name': 'TELEMACH - GLAVNI TRG - STARI PERON', 'bike_stands': 22},
        {'id': 20, 'name': 'LJUBLJANSKA UL. - FOCHEVA', 'bike_stands': 22},
        {'id': 21, 'name': 'VZAJEMNA, VARUH ZDRAVJA - BETNAVSKA C.', 'bike_stands': 22},
        {'id': 22, 'name': 'GOSPOSVETSKA C. - TURNERJEVA UL.', 'bike_stands': 22},
        {'id': 23, 'name': 'PARTIZANSKA C. - TIC', 'bike_stands': 22},
        {'id': 24, 'name': 'MLADINSKA UL. - TRUBARJEVA UL.', 'bike_stands': 22},
        {'id': 25, 'name': 'ULICA MOŠE PIJADA - UKC', 'bike_stands': 22},
        {'id': 26, 'name': 'DVORANA TABOR', 'bike_stands': 22},
        {'id': 27, 'name': 'EUROPARK - POBREŠKA C.', 'bike_stands': 22},
        {'id': 28, 'name': 'PETROL – LENT – VODNI STOLP', 'bike_stands': 22}
    ]

    for station in stations:
        station['name'] = station['name'].replace('.', '').replace(' ', '_')
        if station['id'] == station_id:
            return station

    return None  # Return None if station_id is not found


def main():
    app = Flask(__name__)
    CORS(app)

    @app.route('/api/predict/<int:station_id>', methods=['GET'])
    def predict(station_id):
        station_data = get_station_data(station_id)

        model = tensorflow.keras.models.load_model(ROOT_DIR + '/models/model=' + station_data['name'] + '.h5')
        other_scaler = joblib.load(ROOT_DIR + '/models/scalers/other_scaler=' + station_data['name'] + '.pkl')
        abs_scaler = joblib.load(ROOT_DIR + '/models/scalers/abs_scaler=' + station_data['name'] + '.pkl')
        qt = joblib.load(ROOT_DIR + '/models/transformers/transformer=' + station_data['name'] + '.pkl')

        print(station_data['name'])

        # Get current time rounded to hour
        current_datetime = datetime.now().replace(minute=0, second=0, microsecond=0)
        yesterday_datetime = current_datetime - timedelta(days=1)

        today_file_path = ROOT_DIR + '/data/raw/weather/weather-' + str(current_datetime.date()) + '.csv'
        yesterday_file_path = ROOT_DIR + '/data/raw/weather/weather-' + str(yesterday_datetime.date()) + '.csv'

        # Checks if weather data for this day does not exist yet
        if os.path.isfile(today_file_path):
            df = pd.read_csv(today_file_path)
            df['date'] = pd.to_datetime(df['date'])

            if os.path.isfile(yesterday_file_path):
                df_yesterday = pd.read_csv(yesterday_file_path)
                df_yesterday['date'] = pd.to_datetime(df_yesterday['date'])
                df = pd.concat([df, df_yesterday], ignore_index=True)

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

            columns = columns_of_interest[1:]

            # Fix skewness
            for column in columns:
                array = np.array(df[column]).reshape(-1, 1)
                df[column] = qt.fit_transform(array)

            # Normalize
            df[columns] = other_scaler.fit_transform(df[columns])

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
                "station_name": station_data['name'],
                "station_bike_stands": station_data['bike_stands'],
                "predictions": predictions
            })

        return jsonify({"Message": "Data not available."})

    app.run(host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main()
