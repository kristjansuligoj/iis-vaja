from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import DuplicateKeyError

from datetime import datetime, date
from dotenv import load_dotenv

import os

load_dotenv()

uri = f"mongodb+srv://{os.getenv('MONGODB_USERNAME')}:{os.getenv('MONGODB_PASSWORD')}@iis.vya5pew.mongodb.net/?retryWrites=true&w=majority&appName=iis"


def get_database_client():
    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None


def insert_data(collection_name, data):
    try:
        client = get_database_client()
        if client:
            db = client.get_database('iis')
            collection = db.get_collection(collection_name)
            collection.insert_one(data)
            print("Data inserted successfully!")
    except DuplicateKeyError:
        print("Data with the same _id already exists!")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_predictions_today_by_station(station_name):
    try:
        client = get_database_client()
        if client:
            db = client.get_database('iis')
            collection = db.get_collection('predictions')
            # Define start and end of today
            start_of_day = datetime.combine(date.today(), datetime.min.time())
            end_of_day = datetime.combine(date.today(), datetime.max.time())
            # Query for predictions from today for the specified station
            predictions_today = collection.find({
                'station': station_name,
                'date': {'$gte': start_of_day, '$lte': end_of_day}
            })
            return list(predictions_today)
    except Exception as e:
        print(f"An error occurred: {e}")