import requests
import json


def main():
    url = 'https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b'

    try:
        # Send GET request
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse JSON response
            data = response.json()

            # Save the data to a file
            with open('../../data/raw/unprocessed_data.json', 'w') as f:
                json.dump(data, f, indent=4)  # Write the JSON data to the file with indentation

            print("Data fetched and saved to /data/raw/unprocessed_data.json")
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")

    except requests.RequestException as e:
        # If an error occurs during the request, print the error message
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()