from common.common import save_df_to_csv
from definitions import ROOT_DIR

import pandas as pd


def main():
    try:
        mbajk_df = pd.read_csv(ROOT_DIR + '/data/raw/mbajk/preprocessed_mbajk.csv')
        weather_df = pd.read_csv(ROOT_DIR + '/data/raw/weather/preprocessed_weather.csv')

        mbajk_df['date'] = pd.to_datetime(mbajk_df['date'])
        weather_df['date'] = pd.to_datetime(weather_df['date'])

        # Merge weather data with previous response
        merged_df = pd.merge_asof(mbajk_df, weather_df, on='date')

        # Save a .csv file for each station
        for station_name in merged_df['name'].unique():
            # Drop unnecessary columns
            station_df = merged_df[merged_df['name'] == station_name].drop('name', axis=1)

            station_name = station_name.replace('.', '').replace(',', '').replace(' ', '_')

            # Save the file
            file_path = ROOT_DIR + f'/data/processed/processed_data={station_name}.csv'
            save_df_to_csv(file_path, station_df)

    except FileNotFoundError:
        print("File not found. Please make sure the file exists.")
    except IOError as e:
        print(f"An error occurred while processing the file: {e}")


if __name__ == "__main__":
    main()
