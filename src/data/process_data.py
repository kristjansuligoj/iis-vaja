import pandas as pd
from common.common import save_df_to_csv
import json


def main():
    try:
        # Open the file in read mode
        with open('../../data/raw/mbajk.json', 'r') as f:
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

        # Fetch weather data
        weather_df = pd.read_csv('../../data/raw/weather.csv')

        # Transform date to same format bike_df date is, so merge can happen
        weather_df['date'] = pd.to_datetime(weather_df['date']).dt.tz_localize(None)

        # Sort by last_update column, transform to UNIX datetime, then drop the last_update column
        df.sort_values(by='last_update', inplace=True)
        df['date'] = pd.to_datetime(df['last_update'], unit='ms')
        df.drop('last_update', axis=1, inplace=True)

        # Merge weather data with previous response
        merged_df = pd.merge_asof(df, weather_df, left_on='date', right_on='date')

        # Save a .csv file for each station
        for station_name in merged_df['name'].unique():
            # Drop unnecessary columns
            station_df = merged_df[merged_df['name'] == station_name].drop('name', axis=1)

            # Save the file
            file_path = f'../../data/processed/processed_data_{station_name}.csv'
            save_df_to_csv(file_path, station_df)

    except FileNotFoundError:
        print("File not found. Please make sure the file exists.")
    except IOError as e:
        print(f"An error occurred while processing the file: {e}")


if __name__ == "__main__":
    main()
