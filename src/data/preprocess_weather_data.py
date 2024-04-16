from datetime import datetime, timedelta
from definitions import ROOT_DIR

import pandas as pd
import os


def main():
    try:
        # Read all weather data
        all_file_path = os.path.join(ROOT_DIR, 'data', 'raw', 'weather', 'all-weather.csv')
        all_df = pd.read_csv(all_file_path)
        all_df['date'] = pd.to_datetime(all_df['date'])

        # Read today's weather
        today_file_path = os.path.join(ROOT_DIR, 'data', 'raw', 'weather', 'weather.csv')
        today_df = pd.read_csv(today_file_path)

        # Clears the file for further use
        with open(today_file_path, 'w'):
            pass

        # Filter yesterday's data
        today_date = datetime.now().date()
        yesterday_date = today_date - timedelta(days=1)
        yesterday_df = all_df.loc[all_df['date'].dt.date == yesterday_date]

        # Concatenate today and yesterday's data
        processed_df = pd.concat([yesterday_df, today_df], ignore_index=True)

        # If today weather data does not yet exist, save it
        if not (today_date in all_df['date'].dt.date.unique()):
            all_df = pd.concat([all_df, today_df], ignore_index=True)
            all_df.to_csv(all_file_path, index=False)

        # Save to preprocessed file
        preprocessed_file_path = os.path.join(ROOT_DIR, 'data', 'raw', 'weather', 'preprocessed_weather.csv')
        processed_df.to_csv(preprocessed_file_path, index=False)
    except FileNotFoundError:
        print("File not found. Please make sure the file exists.")
    except IOError as e:
        print(f"An error occurred while processing the file: {e}")


if __name__ == "__main__":
    main()
