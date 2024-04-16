import pytest
import requests


def test_open_meteo_api_endpoint():
    url = 'https://api.open-meteo.com/v1/forecast'

    try:
        # Send GET request
        response = requests.get(url)

        # Check if the request was successful
        response.raise_for_status()

    except requests.RequestException as e:
        pytest.fail(f"Request failed: {e}")

    # Assert that the status code is 200
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}. Response content: {response.content}"
