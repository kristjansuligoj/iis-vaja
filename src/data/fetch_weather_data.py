import requests
import openmeteo_requests
import requests_cache
import pandas as pd
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
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation_probability",
            "rain",
            "showers",
            "snowfall",
            "cloud_cover",
        ],
        "forecast_days": 1,
    }
    response = openmeteo.weather_api(url, params=params)[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(2).ValuesAsNumpy()
    hourly_precipitation_probability = hourly.Variables(3).ValuesAsNumpy()
    hourly_rain = hourly.Variables(4).ValuesAsNumpy()
    hourly_showers = hourly.Variables(5).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(6).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(7).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ), "temperature_2m": hourly_temperature_2m,
        "relative_humidity_2m": hourly_relative_humidity_2m,
        "apparent_temperature": hourly_apparent_temperature,
        "precipitation_probability": hourly_precipitation_probability,
        "rain": hourly_rain,
        "showers": hourly_showers,
        "snowfall": hourly_snowfall,
        "cloud_cover": hourly_cloud_cover
    }

    return pd.DataFrame(data=hourly_data)


def main():
    mb_latitude = 46.555362126247445
    mb_longitude = 15.63425320764157

    try:
        weather_data = fetch_weather_data(mb_latitude, mb_longitude)
        save_df_to_csv('../../data/raw/weather.csv', weather_data)

    except requests.RequestException as e:
        # If an error occurs during the request, print the error message
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()