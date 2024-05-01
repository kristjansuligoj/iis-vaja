import pandas as pd
import numpy as np
import tensorflow
import dagshub
import joblib
import mlflow
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.ensemble import BaggingRegressor

from common.common import make_directory_if_missing
from definitions import ROOT_DIR
from dotenv import load_dotenv

load_dotenv()


def get_metrics(inverse_y_test, inverse_model_predictions):
    mae = mean_absolute_error(inverse_y_test, inverse_model_predictions)
    mse = mean_squared_error(inverse_y_test, inverse_model_predictions)
    evs = explained_variance_score(inverse_y_test, inverse_model_predictions)

    return mae, mse, evs


def split_multifeature_dataset(dataset, window_size=186, num_features=5):
    data_length = len(dataset)
    X = []
    y = []

    for i in range(data_length - window_size):
        window = dataset[i:i + window_size, :]  # V okno vključimo vse značilke
        target = dataset[i + window_size, :]  # Target vključuje vse značilke za naslednji časovni korak
        X.append(window)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    return X, y[:, 0]


def create_gru_model(input_shape):
    model = tensorflow.keras.models.Sequential()

    model.add(tensorflow.keras.layers.GRU(32, return_sequences=True, input_shape=input_shape))

    model.add(tensorflow.keras.layers.GRU(32, activation='relu'))

    model.add(tensorflow.keras.layers.Dense(16, activation='relu'))

    model.add(tensorflow.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def get_model_metrics(model, scaler, predict_X, predict_y):
    # Predict with model
    gru_predictions = model.predict(predict_X)

    # Inverse standardisation
    inverse_gru_predictions = scaler.inverse_transform(gru_predictions)
    inverse_y = scaler.inverse_transform(predict_y.reshape(-1, 1)).flatten()

    # Calculate metrics of TEST data
    return get_metrics(inverse_y, inverse_gru_predictions)


def fill_empty_values(df):
    # Identify columns with missing values
    columns_with_missing_values = df.columns[df.isnull().any()].tolist()
    columns_with_complete_values = df.drop(
        columns_with_missing_values + ["date"] + ["address"],
        axis=1
    ).columns.tolist()

    # Create DataFrames for missing and complete data
    missing_df = df[df.isnull().any(axis=1)]
    complete_df = df.dropna()

    for column in columns_with_missing_values:
        X = complete_df[columns_with_complete_values]
        y = complete_df[column]

        # Train the model
        model = BaggingRegressor()
        model.fit(X, y)

        # Make predictions
        missing_X = missing_df[columns_with_complete_values]
        predictions = model.predict(missing_X)

        # Fill missing values in original DataFrame
        df.loc[missing_df.index, column] = predictions

    return df


def main():
    station_directory = ROOT_DIR + '/data/processed/'

    station_names = []

    for filename in os.listdir(station_directory):
        if filename.startswith('processed_data='):
            # Parse station name from the file name
            station_name = filename.split('=')[1].split('.')[0]
            # Append station name to the list
            station_names.append(station_name)

    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(os.getenv("DAGSHUB_REPO_NAME"), os.getenv("DAGSHUB_USERNAME"), mlflow=True)
    mlflow.set_tracking_uri(os.getenv("DAGSHUB_URI"))

    for station_name in station_names:
        mlflow.start_run(run_name=f"run={station_name}")
        mlflow.tensorflow.autolog()

        window_size = 24
        batch_size = 32
        epochs = 10

        # Read data
        df_train = pd.read_csv(station_directory + 'train_data=' + station_name + '.csv')
        df_test = pd.read_csv(station_directory + 'test_data=' + station_name + '.csv')

        # Sort by date
        df_train['date'] = pd.to_datetime(df_train['date'])
        df_test['date'] = pd.to_datetime(df_test['date'])

        df_train = df_train.sort_values('date')
        df_test = df_test.sort_values('date')

        df_train = fill_empty_values(df_train)
        df_test = fill_empty_values(df_test)

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

        df_train_target_columns = df_train[columns_of_interest].values
        df_test_target_columns = df_test[columns_of_interest].values
        columns = columns_of_interest[1:]

        qt = QuantileTransformer(n_quantiles=32, output_distribution='normal')

        for column in columns:
            train_array = np.array(df_train[column]).reshape(-1, 1)
            test_array = np.array(df_test[column]).reshape(-1, 1)

            df_train[column] = qt.fit_transform(train_array)
            df_test[column] = qt.fit_transform(test_array)

        # Create directory for models if it doesn't exist
        make_directory_if_missing(os.path.join(ROOT_DIR, "models", "transformers"))

        joblib.dump(qt, ROOT_DIR + '/models/transformers/transformer=' + station_name + '.pkl')

        train_data = df_train_target_columns
        test_data = df_test_target_columns

        abs_train_data = train_data[:, 0]
        abs_test_data = test_data[:, 0]

        other_train_data = train_data[:, 1:]
        other_test_data = test_data[:, 1:]

        # Normalization
        other_scaler = MinMaxScaler()
        normalized_other_train_data = other_scaler.fit_transform(other_train_data)
        normalized_other_test_data = other_scaler.transform(other_test_data)

        make_directory_if_missing(os.path.join(ROOT_DIR, "models", "scalers"))

        joblib.dump(
            other_scaler,
            ROOT_DIR + '/models/scalers/other_scaler=' + station_name + '.pkl'
        )

        abs_scaler = MinMaxScaler()
        normalized_abs_train_data = abs_scaler.fit_transform(abs_train_data.reshape(-1, 1))
        normalized_abs_test_data = abs_scaler.transform(abs_test_data.reshape(-1, 1))

        joblib.dump(
            abs_scaler,
            ROOT_DIR + '/models/scalers/abs_scaler=' + station_name + '.pkl'
        )

        normalized_train_data = np.column_stack([
            normalized_other_train_data,
            normalized_abs_train_data
        ])

        normalized_test_data = np.column_stack([
            normalized_other_test_data,
            normalized_abs_test_data
        ])

        # Split the data
        X_train, y_train = split_multifeature_dataset(normalized_train_data, window_size=window_size)
        X_test, y_test = split_multifeature_dataset(normalized_test_data, window_size=window_size)

        X_train = np.transpose(X_train, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))

        input_shape = (X_train.shape[1], X_train.shape[2])

        # Create the model
        gru_model = create_gru_model(input_shape)

        # Fit the model for specific staiton
        gru_model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size)

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("train_dataset_size", len(train_data))

        # Save the model for specific station
        gru_model.save(ROOT_DIR + '/models/model=' + station_name + '.h5')

        # Get metrics of TEST data
        gru_mae, gru_mse, gru_evs = get_model_metrics(gru_model, abs_scaler, X_test, y_test)

        mlflow.log_metric("mae", gru_mae)
        mlflow.log_metric("mse", gru_mse)
        mlflow.log_metric("evs", gru_evs)

        make_directory_if_missing(os.path.join(ROOT_DIR, "reports"))
        with open(ROOT_DIR + '/reports/model=' + station_name + '-metrics.txt', 'w', encoding='utf-8') as f:
            f.write(
                f'Mean average error: {gru_mae}\nMean square error: {gru_mse}\nExplained variance score: {gru_evs}\n')

        # Get metrics of TRAIN data
        gru_mae, gru_mse, gru_evs = get_model_metrics(gru_model, abs_scaler, X_train, y_train)

        with open(ROOT_DIR + '/reports/model=' + station_name + '-train_metrics.txt', 'w', encoding='utf-8') as f:
            f.write(
                f'Mean average error: {gru_mae}\nMean square error: {gru_mse}\nExplained variance score: {gru_evs}\n')

        mlflow.end_run()


if __name__ == "__main__":
    main()
