import json
import pandas as pd


def main():
    try:
        # Open the file in read mode
        with open('../../data/raw/unprocessed_data.json', 'r') as f:
            data = json.load(f)

        # Transform json to csv
        df = pd.DataFrame(data)

        # Only use the columns we need
        columns = [
            'number',
            'name',
            'address',
            'position',
            'bike_stands',
            'available_bike_stands',
            'available_bikes',
            'last_update'
        ]
        df = df[columns]

        for station_name in df['name'].unique():
            station_df = df[df['name'] == station_name].drop('name', axis=1)

            # Transform UNIX ms to date time
            station_df['date'] = pd.to_datetime(station_df['last_update'], unit='ms')
            station_df = station_df.drop('last_update', axis=1)

            processed_file_name = f'../../data/processed/processed_data_{station_name}.csv'
            station_df.to_csv(processed_file_name, mode='a', index=False, header=False)

    except FileNotFoundError:
        print("File not found. Please make sure the file exists.")
    except IOError as e:
        print(f"An error occurred while processing the file: {e}")


if __name__ == "__main__":
    main()
