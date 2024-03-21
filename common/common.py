import os
import json


def save_df_to_csv(file_path, df):
    # Checks if file exists, so it knows whether the csv file should add headers or not
    headers = True
    if os.path.isfile(file_path):
        headers = False

    # Save file
    df.to_csv(file_path, mode='a', index=False, header=headers)
    print(f"File saved to {file_path}")


def save_to_json(file_path, data):
    # Load existing data if file exists
    try:
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []

    # Append new data to existing data
    existing_data.append(data)

    # Save the combined data to the file
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)  # Write the combined JSON data to the file with indentation

    print(f"Data appended and saved to {file_path}")
