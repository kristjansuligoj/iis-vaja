from flask import Flask, jsonify, request

import tensorflow
import pandas as pd
import numpy as np
import joblib

model = tensorflow.keras.models.load_model('src/models/base_data_model.h5')
scaler = joblib.load('src/models/base_data_scaler.pkl')

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

    # Validate JSON data before request handling
    @app.before_request
    def validate_json():
        if request.endpoint == 'mbajk/predict':
            # Validate JSON structure and keys for endpoint mbajk/predict
            if len(request.json) != 186:
                error_message = "Error! Expected 186 data points with 'date' and 'available_bike_stands' information."
                return jsonify({"error": error_message}), 400
            for entry in request.json:
                keys = list(entry.keys())
                if keys != expected_structure:
                    error_message = f"Entry {entry} does not match the expected structure."
                    return jsonify({"error": error_message}), 400

    @app.route('/mbajk/predict', methods=['POST'])
    def predict():
        # Convert JSON data to a DataFrame
        df = pd.DataFrame(request.json)

        # Convert 'date' column to datetime format and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Transform data and make prediction
        available_bike_stands = df['available_bike_stands'].values.reshape(-1, 1)
        n_available_bike_stands = scaler.transform(available_bike_stands)
        n_available_bike_stands = np.reshape(
            n_available_bike_stands,
            (n_available_bike_stands.shape[1], 1, n_available_bike_stands.shape[0])
        )

        # Make predictions using model
        prediction = model.predict(n_available_bike_stands)

        # Inverse the transformation
        inversed_n_bike_stands = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))

        return jsonify({'prediction': inversed_n_bike_stands.tolist()})

    app.run(host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main()
