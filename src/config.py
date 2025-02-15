import os

from dotenv import load_dotenv

from src.feature_store_api import FeatureGroupConfig, FeatureViewConfig


from src.paths import PARENT_DIR

# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / '.env')

HOPSWORKS_PROJECT_NAME = "NYC_taxi_demand_project"
try:
    # HOPSWORKS_PROJECT_NAME = os.environ['HOPSWORKS_PROJECT_NAME']
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception(
        'Create an .env file on the project root with the HOPSWORKS_API_KEY'
    )

MODEL_NAME = 'taxi_demand_predictor_next_hour'
MODEL_VERSION = '1'

# # TODO: remove FEATURE_GROUP_NAME and FEATURE_GROUP_VERSION, and use FEATURE_GROUP_METADATA instead
FEATURE_GROUP_NAME = "time_series_hourly_feature_group"
FEATURE_GROUP_VERSION = 1 # updated to incldude pickup_hour as primary ket. V3 does not

FEATURE_GROUP_METADATA = FeatureGroupConfig(
    name='time_series_hourly_feature_group',
    version=3,
    description='Feature group with hourly time-series data of historical taxi rides',
    primary_key=['pickup_location_id', 'pickup_ts'],
    event_time='pickup_ts',
    online_enabled=True,
)

# TODO: remove FEATURE_VIEW_NAME and FEATURE_VIEW_VERSION, and use FEATURE_VIEW_METADATA instead
FEATURE_VIEW_NAME = 'time_series_hourly_feature_view'
FEATURE_VIEW_VERSION = 4

FEATURE_VIEW_METADATA = FeatureViewConfig(
    name='time_series_hourly_feature_view',
    version=4,
    feature_group=FEATURE_GROUP_METADATA,
)

# added for monitoring purposes
# TODO remove FEATURE_GROUP_MODEL_PREDICTIONS and use FEATURE_GROUP_PREDICTIONS_METADATA instead
FEATURE_GROUP_MODEL_PREDICTIONS = 'model_predictions_feature_group'
FEATURE_GROUP_PREDICTIONS_METADATA = FeatureGroupConfig(
    name='model_predictions_feature_group',
    version=1,
    description='Predictions generate by our production model',
    primary_key=['pickup_location_id', 'pickup_ts'],
    event_time='pickup_ts'
    )

# TODO remove FEATURE_VIEW_MODEL_PREDICTIONS and use FEATURE_VIEW_PREDICTIONS_METADATA instead
FEATURE_VIEW_MODEL_PREDICTIONS = 'model_predictions_feature_view'
FEATURE_VIEW_PREDICTIONS_METADATA = FeatureViewConfig(
    name='model_predictions_feature_view',
    version=4,
    feature_group=FEATURE_GROUP_PREDICTIONS_METADATA,
)

# Number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28 * 1 

# Number of iterations we want Optuna to perform to find the best hyperparameters
N_HYPERPARAMETER_SEARCH_TRIALS = 1

# Maximum Mean Absolute Error we allow our production model to have
MAX_MAE = 30.0

MONITORING_FV_NAME = 'monitoring_feature_view'
MONITORING_FV_VERSION = 4


# Debugging...!
# %%
# %%
# def inspect_feature_group():
#     from src.feature_store_api import get_feature_store
#     import src.config as config
    
#     fs = get_feature_store()
#     fg = fs.get_feature_group(
#         name=config.FEATURE_GROUP_MODEL_PREDICTIONS,  # replace with your actual feature group name
#         version=1
#     )
    
#     # Get the schema
#     print("Feature group schema:")
#     print(fg.schema)
    
#     return fg.schema
# inspect_feature_group()
# %%

# from datetime import datetime, timedelta
# import pandas as pd 
# from src.feature_store_api import get_or_create_feature_view
# def investigate_time_series_windows():
#     """
#     Investigates how different time windows affect our data collection, with special
#     attention to the final prediction hour (23:00-24:00).
#     """
#     # Set up our reference time - we want to predict the 23:00-24:00 hour
#     current_date = pd.to_datetime('2024-10-30 23:00:00')
#     feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)
    
#     print("=== Data Window Analysis ===")
#     # Test 1: Including the full final hour (until 24:00)
#     data_complete = feature_view.get_batch_data(
#         start_time=current_date - timedelta(days=28),
#         end_time=current_date + timedelta(hours=1)  # Until 24:00
#     )
    
#     # Test 2: Original window (until 23:00)
#     data_incomplete = feature_view.get_batch_data(
#         start_time=current_date - timedelta(days=28),
#         end_time=current_date
#     )
    
#     print(f"Records with complete final hour (until 24:00): {len(data_complete)}")
#     print(f"Records stopping at 23:00: {len(data_incomplete)}")
#     print(f"Difference: {len(data_complete) - len(data_incomplete)}")
    
#     # Analyze the final hour specifically
#     final_hour_records = data_complete[
#         data_complete['pickup_hour'] == current_date
#     ]
#     print(f"\nRecords specifically for 23:00-24:00 period: {len(final_hour_records)}")
    
#     if len(final_hour_records) > 0:
#         print("\nSample of final hour records:")
#         print(final_hour_records.head())
    
#     # Check total hours per location
#     hours_per_location = data_complete.groupby('pickup_location_id')['pickup_hour'].nunique()
#     print(f"\nTypical hours per location: {hours_per_location.value_counts().head()}")

#     return data_complete, final_hour_records

# # Run the investigation
# # complete_data, final_hour = investigate_time_series_windows()

# # %%
# import src.config as config
# import numpy as np
# current_date = pd.to_datetime('2024-10-30 23:00:00')
# def load_batch_of_features_from_store(current_date: pd.Timestamp) -> pd.DataFrame:
#     fetch_data_from = (current_date - timedelta(days=28)).tz_localize(None)
#     fetch_data_to = (current_date + timedelta(hours=1)).tz_localize(None)
    
#     feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)
#     ts_data = feature_view.get_batch_data(
#         start_time=fetch_data_from,
#         end_time=fetch_data_to
#     )

#     # Sort and process data without filtering
#     ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)
    
#     # Create feature matrix keeping all hours
#     location_ids = ts_data['pickup_location_id'].unique()
#     x = np.ndarray(shape=(len(location_ids), config.N_FEATURES), dtype=np.float32)
    
#     for i, location_id in enumerate(location_ids):
#         ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
#         ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
#         x[i, :] = ts_data_i['rides'].values[:config.N_FEATURES]

#     features = pd.DataFrame(
#         x,
#         columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(config.N_FEATURES))]
#     )
#     features['pickup_hour'] = current_date
#     features['pickup_location_id'] = location_ids
    
#     return features
# # load_batch_of_features_from_store(current_date)
    
# # %%
# def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
#     """
#     Adds one column with the average rides from:
#     - 7 days ago
#     - 14 days ago
#     - 21 days ago
#     - 28 days ago
#     """
#     # Ensure X is a DataFrame
#     if not isinstance(X, pd.DataFrame):
#         raise ValueError("Input X must be a pandas DataFrame")
    
#     # List of required columns
#     required_columns = [
#         f'rides_previous_{7*24}_hour',
#         f'rides_previous_{2*7*24}_hour',
#         f'rides_previous_{3*7*24}_hour',
#         f'rides_previous_{4*7*24}_hour',
#     ]

#     # Check for missing columns
#     for col in required_columns:
#         if col not in X.columns:
#             X[col] = 0  # Or handle differently based on use case
    
#     # Compute the average
#     X['average_rides_last_4_weeks'] = 0.25 * (
#         X[f'rides_previous_{7*24}_hour'] +
#         X[f'rides_previous_{2*7*24}_hour'] +
#         X[f'rides_previous_{3*7*24}_hour'] +
#         X[f'rides_previous_{4*7*24}_hour']
#     )
    
#     return X

# data = {
#     f'rides_previous_{7*24}_hour': [10, 20],
#     f'rides_previous_{2*7*24}_hour': [15, 25],
#     f'rides_previous_{3*7*24}_hour': [20, 30],
#     f'rides_previous_{4*7*24}_hour': [25, 35],
# }
# X = pd.DataFrame(data)
# #result = average_rides_last_4_weeks(X)
# #print(result)
# # %%
# from src.inference import load_batch_of_features_from_store, load_model_from_registry, get_model_predictions

# current_date = pd.to_datetime('2024-10-30 23:00:00')
# fetch_data_from = (current_date - timedelta(days=28)).tz_localize(None)
# fetch_data_to = (current_date + timedelta(hours=1)).tz_localize(None)

# print(fetch_data_from)
# print(fetch_data_to)
# # %%
# feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)
# ts_data = feature_view.get_batch_data(
#     start_time=fetch_data_from,
#     end_time=fetch_data_to
# )

# print(feature_view)
# print(ts_data.head())
# # %%
# # Sort and process data without filtering
# ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)

# # Create feature matrix keeping all hours
# location_ids = ts_data['pickup_location_id'].unique()
# x = np.ndarray(shape=(len(location_ids), config.N_FEATURES), dtype=np.float32)

# print(location_ids.shape)
# print(x.shape)
# # %%
# # for i, location_id in enumerate(location_ids):
# #     ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
# #     ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
# #     x[i, :] = ts_data_i['rides'].values[:config.N_FEATURES]
# for i, location_id in enumerate(location_ids):
#     ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
#     ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
    
#     # Debugging output to identify the issue
#     rides_values = ts_data_i['rides'].values
#     print(f"Location ID: {location_id}")
#     print(f"Number of data points: {len(rides_values)}")
#     print(f"Expected: {config.N_FEATURES}")
    
#     if len(rides_values) < config.N_FEATURES:
#         print(f"** Insufficient data for location_id {location_id}. Skipping or handling required. **")
#         continue  # Skip this iteration for now
    
#     if len(rides_values) > config.N_FEATURES:
#         print(f"** More data than expected for location_id {location_id}. Truncating to fit. **")
    
#     # Assignment (will trigger an error if len(rides_values) < config.N_FEATURES)
#     x[i, :] = rides_values[:config.N_FEATURES]



# print(ts_data.head())
# print(x)
# # %%
# print(type(x))
# print(x)
# # %%
# features = pd.DataFrame(
#         x,
#         columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(config.N_FEATURES))]
#     )
# features['pickup_hour'] = current_date
# features['pickup_location_id'] = location_ids

# print(features)
# print(type(features))
# # %%
# # %%
# # model = load_model_from_registry()

# import joblib
# from pathlib import Path
# import src.config as config
# import hopsworks
# def get_hopsworks_project() -> hopsworks.project.Project:

#     return hopsworks.login(
#         project=config.HOPSWORKS_PROJECT_NAME,
#         api_key_value=config.HOPSWORKS_API_KEY
#     )
# # %% 
# project = get_hopsworks_project()
# model_registry = project.get_model_registry()

# # print(project)
# print(model_registry)
# # %%
# model = model_registry.get_model(
#     name = "taxi_demand_predictor_next_hour",
#     version=config.MODEL_VERSION,
# )  
# print(model)

# # %% 
# model_dir = model.download()
# model = joblib.load(Path(model_dir)  / 'model.pkl')
# print(model)
# # %%
# results = get_model_predictions(model, features)
# print(results)
# # %%
