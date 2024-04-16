import pytest
import requests


def test_mbajk_api_endpoint():
    url = 'https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b'

    try:
        # Send GET request
        response = requests.get(url)

        # Check if the request was successful
        response.raise_for_status()

    except requests.RequestException as e:
        pytest.fail(f"Request failed: {e}")

    # Assert that the status code is 200
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}. Response content: {response.content}"
