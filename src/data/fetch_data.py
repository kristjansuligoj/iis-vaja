import requests
from common.common import save_to_json


def main():
    url = 'https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b'

    try:
        # Send GET request
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse JSON response
            mbajk_data = response.json()

            save_to_json('../../data/raw/mbajk.json', mbajk_data)
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")

    except requests.RequestException as e:
        # If an error occurs during the request, print the error message
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()