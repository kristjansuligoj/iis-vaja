from common.common import save_to_json

import requests
import openmeteo_requests
import requests_cache
import pandas as pd
from datetime import datetime
from definitions import ROOT_DIR
from common.common import save_df_to_csv
from retry_requests import retry


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
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "is_day",
            "precipitation",
            "rain",
            "showers",
            "snowfall",
            "cloud_cover",
            "wind_speed_10m",
        ],
        "forecast_days": 1,
    }
    response = openmeteo.weather_api(url, params=params)[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    current = response.Current()
    current_temperature_2m = current.Variables(0).Value()
    current_relative_humidity_2m = current.Variables(1).Value()
    current_apparent_temperature = current.Variables(2).Value()
    current_is_day = current.Variables(3).Value()
    current_precipitation = current.Variables(4).Value()
    current_rain = current.Variables(5).Value()
    current_showers = current.Variables(6).Value()
    current_snowfall = current.Variables(7).Value()
    current_cloud_cover = current.Variables(8).Value()
    current_wind_speed_10m = current.Variables(9).Value()

    weather_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature_2m": current_temperature_2m,
        "relative_humidity_2m": current_relative_humidity_2m,
        "apparent_temperature": current_apparent_temperature,
        "is_day": current_is_day,
        "precipitation": current_precipitation,
        "rain": current_rain,
        "showers": current_showers,
        "snowfall": current_snowfall,
        "cloud_cover": current_cloud_cover,
        "wind_speed_10m": current_wind_speed_10m,
    }

    return pd.DataFrame([weather_data])


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
                coordinates = mbajk_data[0]['position']
                weather_data = fetch_weather_data(coordinates['lat'], coordinates['lng'])
                file_path = ROOT_DIR + '/data/raw/weather.csv'
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