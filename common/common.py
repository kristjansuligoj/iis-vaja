import os
import json


def save_df_to_csv(file_path, df):
    df.to_csv(file_path, mode='a', index=False)
    print(f"File saved to {file_path}")


def save_to_json(file_path, data):
    # Load existing data if file exists
    try:
        if os.path.getsize(file_path) > 0:
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
    except FileNotFoundError:
        existing_data = []

    # Append new data to existing data
    existing_data.extend(data)

    # Save the combined data to the file
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)  # Write the combined JSON data to the file with indentation

    print(f"Data appended and saved to {file_path}")
