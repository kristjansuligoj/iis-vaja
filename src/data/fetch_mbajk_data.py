from common.common import save_to_json
from definitions import ROOT_DIR

import requests

def main():
    url = 'https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b'

    try:
        # Send GET request
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse JSON response
            mbajk_data = response.json()

            file_path = ROOT_DIR + '/data/raw/mbajk/mbajk.json'
            save_to_json(file_path, mbajk_data)
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")

    except requests.RequestException as e:
        # If an error occurs during the request, print the error message
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
