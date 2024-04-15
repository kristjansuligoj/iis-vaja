from definitions import ROOT_DIR
from datetime import datetime, timedelta

import pandas as pd


def main():
    try:
        file_path = ROOT_DIR + '/data/raw/weather/by-date/weather='

        today_df = pd.read_csv(file_path + str(datetime.now().date()) + '.csv')
        yesterday_df = pd.read_csv(file_path + str(datetime.now().date() - timedelta(days=1)) + '.csv')

        df = pd.concat([today_df, yesterday_df], ignore_index=True)
        df.sort_values(by='date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        file_path = '../../data/raw/weather/preprocessed_weather.csv'
        df.to_csv(file_path, index=False)
    except FileNotFoundError:
        print("File not found. Please make sure the file exists.")
    except IOError as e:
        print(f"An error occurred while processing the file: {e}")


if __name__ == "__main__":
    main()
