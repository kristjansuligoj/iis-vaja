import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
import tensorflow


def get_metrics(inverse_y_test, inverse_model_predictions):
    mae = mean_absolute_error(inverse_y_test, inverse_model_predictions)
    mse = mean_squared_error(inverse_y_test, inverse_model_predictions)
    evs = explained_variance_score(inverse_y_test, inverse_model_predictions)

    return mae, mse, evs


def split_dataset(dataset, window_size=186):
    data_length = len(dataset)
    X = []
    y = []

    for i in range(data_length - window_size):
        window = dataset[i:i + window_size, 0]
        target = dataset[i + window_size, 0]
        X.append(window)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    return np.reshape(X, (X.shape[0], 1, X.shape[1])), y


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


def main():
    window_size = 186

    # Read data
    df = pd.read_csv('../../data/raw/mbajk_dataset.csv')

    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Take only "available_bike_stands" column
    available_bike_stands_column = df['available_bike_stands'].values.reshape(-1, 1)

    # Train length is all of available data
    total_length = len(available_bike_stands_column)
    test_length = 1302 + window_size
    train_length = total_length - test_length

    train_data = available_bike_stands_column[:train_length]
    test_data = available_bike_stands_column[train_length:]

    # Normalization
    scaler = MinMaxScaler()
    normalized_train_data = scaler.fit_transform(train_data)
    normalized_test_data = scaler.transform(test_data)

    # Save the scaler
    joblib.dump(scaler, 'base_data_scaler.pkl')

    # Split the data
    X_train, y_train = split_dataset(normalized_train_data, window_size=window_size)
    X_test, y_test = split_dataset(normalized_test_data, window_size=window_size)

    input_shape = (X_train.shape[1], X_train.shape[2])

    # Create the model
    gru_model = create_gru_model(input_shape)

    # Fit the model
    gru_model.fit(X_train, y_train, epochs=5, validation_split=0.2)

    # Save the model
    gru_model.save('./base_data_model.h5')

    # Get metrics of TEST data
    gru_mae, gru_mse, gru_evs = get_model_metrics(gru_model, scaler, X_test, y_test)

    with open('../../reports/metrics.txt', 'w', encoding='utf-8') as f:
        f.write(f'Mean average error: {gru_mae}\nMean square error: {gru_mse}\nExplained variance score: {gru_evs}\n')

    # Get metrics of TRAIN data
    gru_mae, gru_mse, gru_evs = get_model_metrics(gru_model, scaler, X_train, y_train)

    with open('../../reports/train_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(f'Mean average error: {gru_mae}\nMean square error: {gru_mse}\nExplained variance score: {gru_evs}\n')


if __name__ == "__main__":
    main()
