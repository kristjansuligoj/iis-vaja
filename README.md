# IIS - MBajk prediction app

## Description
This app uses a RNN model to predict available bike stands at a specific stop, based on data on previous days.

## Requirements
- Poetry
- Python >= 3.9

## Dependencies

This project uses [Poetry](https://python-poetry.org/) for dependency management. Poetry takes care of installing all the necessary dependencies for running the application.

## Installation
1. Make sure you have Poetry installed. You can check by running the following command:
    ```
    poetry --version
    ```
   If not installed, follow the [Poetry installation guide](https://python-poetry.org/docs/#installation) to install Poetry.

2. Clone this repository:
    ```
    git clone https://github.com/kristjansuligoj/iis-vaja.git
    ```

3. Navigate to the project directory:
    ```
    cd iis-vaja
    ```

4. Install project dependencies:
    ```
    poetry install
    ```

## Usage
To run the API, use the following command:
```
poetry run api
```


Once the API is running, it exposes an endpoint at `http://0.0.0.0:5000/predict/mbajk`.

### Making Predictions

To get a prediction, send a POST request to the endpoint with previous data points. You can use tools like cURL, Postman, or any HTTP client library.

### Testing the API

An example of the raw body that should be sent for testing the API can be found in `/data/test/test-api-data.json`. You can use this sample data to verify that the API is functioning correctly.
