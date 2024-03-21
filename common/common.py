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
    # Save the data to a file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)  # Write the JSON data to the file with indentation

    print(f"File saved to {file_path}")
