import os
from datetime import datetime, timedelta
import pandas as pd

from definitions import ROOT_DIR  # Assuming ROOT_DIR is defined in definitions.py

def main():
    try:
        # Read all weather data
        all_file_path = os.path.join(ROOT_DIR, 'data', 'raw', 'weather', 'all-weather.csv')
        all_df = pd.read_csv(all_file_path)
        all_df['date'] = pd.to_datetime(all_df['date'])

        # Read today's data
        today_file_path = os.path.join(ROOT_DIR, 'data', 'raw', 'weather', 'weather.csv')
        today_df = pd.read_csv(today_file_path)
        today_df.to_csv(all_file_path, mode='a', header=False, index=False)

        # Filter yesterday's data
        yesterday_df = all_df.loc[all_df['date'].dt.date == (datetime.now().date() - timedelta(days=1))]

        # Concatenate today and yesterday's data
        df = pd.concat([yesterday_df, today_df], ignore_index=True)

        # Save to preprocessed file
        preprocessed_file_path = os.path.join(ROOT_DIR, 'data', 'raw', 'weather', 'preprocessed_weather.csv')
        df.to_csv(preprocessed_file_path, index=False)
    except FileNotFoundError:
        print("File not found. Please make sure the file exists.")
    except IOError as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()
