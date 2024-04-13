from common.common import save_to_json, save_df_to_csv
from definitions import ROOT_DIR
from retry_requests import retry
from datetime import datetime

import openmeteo_requests
import requests_cache
import pandas as pd
import requests
import os


def fetch_weather_data(latitude, longitude):
    # Set up the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation_probability",
            "rain",
            "surface_pressure",
        ],
        "forecast_days": 1,
    }
    response = openmeteo.weather_api(url, params=params)[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
    hourly_precipitation_probability = hourly.Variables(4).ValuesAsNumpy()
    hourly_rain = hourly.Variables(5).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(6).ValuesAsNumpy()

    weather_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
        "temperature": hourly_temperature_2m,
        "relative_humidity": hourly_relative_humidity_2m,
        "dew_point": hourly_dew_point_2m,
        "apparent_temperature": hourly_apparent_temperature,
        "precipitation_probability": hourly_precipitation_probability,
        "rain": hourly_rain,
        "surface_pressure": hourly_surface_pressure
    }

    return pd.DataFrame(weather_data)


def main():
    url = 'https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b'

    try:
        # Send GET request
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse JSON response
            mbajk_data = response.json()

            file_path = ROOT_DIR + '/data/raw/mbajk.json'
            save_to_json(file_path, mbajk_data)

            try:
                file_path = ROOT_DIR + '/data/raw/weather/weather-' + str(datetime.now().date()) + '.csv'

                # Checks if weather data for this day does not exist yet
                if not os.path.isfile(file_path):
                    coordinates = mbajk_data[0]['position']
                    weather_data = fetch_weather_data(coordinates['lat'], coordinates['lng'])
                    save_df_to_csv(file_path, weather_data)

            except requests.RequestException as e:
                print(f"An error occurred: {e}")
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")

    except requests.RequestException as e:
        # If an error occurs during the request, print the error message
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
