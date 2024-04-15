from common.common import save_to_json
from definitions import ROOT_DIR

import pandas as pd
import json


def main():
    try:
        # Open the file in read mode
        with open(ROOT_DIR + '/data/raw/mbajk/mbajk.json', 'r') as f:
            data = json.load(f)

        # Clear the file
        with open(ROOT_DIR + '/data/raw/mbajk/mbajk.json', 'w') as f:
            f.write('')

        save_to_json(ROOT_DIR + '/data/raw/mbajk/all-mbajk.json', data)

        # Transform json to csv
        df = pd.DataFrame(data)

        # Sort by last_update column, transform to UNIX datetime, then drop the last_update column
        df.sort_values(by='last_update', inplace=True)
        df['date'] = pd.to_datetime(df['last_update'], unit='ms')
        df = df[[
            'date',
            'name',
            'address',
            'bike_stands',
            'available_bike_stands',
        ]]

        file_path = ROOT_DIR + '/data/raw/mbajk/preprocessed_mbajk.csv'
        df.to_csv(file_path, index=False)
    except FileNotFoundError:
        print("File not found. Please make sure the file exists.")
    except IOError as e:
        print(f"An error occurred while processing the file: {e}")


if __name__ == "__main__":
    main()
