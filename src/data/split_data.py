from definitions import ROOT_DIR
import pandas as pd
import os


def main():
    stations_directory = os.path.join(ROOT_DIR, 'data', 'processed')

    station_names = []
    for filename in os.listdir(stations_directory):
        if filename.endswith('.csv'):
            # Parse station name from the file name
            station_name = filename.split('=')[1].split('.')[0]
            # Append station name to the list
            station_names.append(station_name)

    for station_name in station_names:
        station_directory = os.path.join(stations_directory, f'processed_data={station_name}.csv')

        station_data = pd.read_csv(station_directory)

        # Sort data by date
        station_data = station_data.sort_values(by="date", ascending=False)

        # Get test size
        test_size = int(len(station_data) * 0.1)

        # Split to train/test data
        train_data = station_data.iloc[test_size:]
        test_data = station_data.iloc[:test_size]

        # Save train/test data
        train_data.to_csv(os.path.join(stations_directory, f"train_data={station_name}.csv"), index=False)
        test_data.to_csv(os.path.join(stations_directory, f"test_data={station_name}.csv"), index=False)


if __name__ == "__main__":
    main()
