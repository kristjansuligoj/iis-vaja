import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor


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


def split_multifeature_dataset(dataset, window_size=186, num_features=5):
    data_length = len(dataset)
    X = []
    y = []

    for i in range(data_length - window_size):
        window = dataset[i:i + window_size, :]
        target = dataset[i + window_size, :]
        X.append(window)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    return X, y[:, 0]


def prepare_train_test_data(station_directory, station_name, window_size, abs_scaler, other_scaler):
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

    train_data = df_train_target_columns
    test_data = df_test_target_columns

    abs_train_data = train_data[:, 0]
    abs_test_data = test_data[:, 0]

    other_train_data = train_data[:, 1:]
    other_test_data = test_data[:, 1:]

    # Normalization
    normalized_other_train_data = other_scaler.fit_transform(other_train_data)
    normalized_other_test_data = other_scaler.transform(other_test_data)

    normalized_abs_train_data = abs_scaler.fit_transform(abs_train_data.reshape(-1, 1))
    normalized_abs_test_data = abs_scaler.transform(abs_test_data.reshape(-1, 1))

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

    return X_train, y_train, X_test, y_test