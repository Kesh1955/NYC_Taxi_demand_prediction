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

# # TODO: remove FEATURE_GROUP_NAME and FEATURE_GROUP_VERSION, and use FEATURE_GROUP_METADATA instead
FEATURE_GROUP_NAME = "time_series_hourly_feature_group"
FEATURE_GROUP_VERSION = 3

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
    version=4,
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


MODEL_NAME = 'taxi_demand_predictor_next_hour'
MODEL_VERSION = '1'


# Number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28 * 1 

# Number of iterations we want Optuna to perform to find the best hyperparameters
N_HYPERPARAMETER_SEARCH_TRIALS = 1

# Maximum Mean Absolute Error we allow our production model to have
MAX_MAE = 30.0

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
N_FEATURES = 24 * 28 * 1 
print(N_FEATURES)

# %%
from datetime import datetime, timedelta
import pandas as pd 
from src.feature_store_api import get_or_create_feature_view
def investigate_time_series_windows():
    """
    Investigates how different time windows affect our data collection, with special
    attention to the final prediction hour (23:00-24:00).
    """
    # Set up our reference time - we want to predict the 23:00-24:00 hour
    current_date = pd.to_datetime('2024-10-30 23:00:00')
    feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)
    
    print("=== Data Window Analysis ===")
    # Test 1: Including the full final hour (until 24:00)
    data_complete = feature_view.get_batch_data(
        start_time=current_date - timedelta(days=28),
        end_time=current_date + timedelta(hours=1)  # Until 24:00
    )
    
    # Test 2: Original window (until 23:00)
    data_incomplete = feature_view.get_batch_data(
        start_time=current_date - timedelta(days=28),
        end_time=current_date
    )
    
    print(f"Records with complete final hour (until 24:00): {len(data_complete)}")
    print(f"Records stopping at 23:00: {len(data_incomplete)}")
    print(f"Difference: {len(data_complete) - len(data_incomplete)}")
    
    # Analyze the final hour specifically
    final_hour_records = data_complete[
        data_complete['pickup_hour'] == current_date
    ]
    print(f"\nRecords specifically for 23:00-24:00 period: {len(final_hour_records)}")
    
    if len(final_hour_records) > 0:
        print("\nSample of final hour records:")
        print(final_hour_records.head())
    
    # Check total hours per location
    hours_per_location = data_complete.groupby('pickup_location_id')['pickup_hour'].nunique()
    print(f"\nTypical hours per location: {hours_per_location.value_counts().head()}")

    return data_complete, final_hour_records

# Run the investigation
# complete_data, final_hour = investigate_time_series_windows()

# %%
import src.config as config
import numpy as np
current_date = pd.to_datetime('2024-10-30 23:00:00')
def load_batch_of_features_from_store(current_date: pd.Timestamp) -> pd.DataFrame:
    fetch_data_from = (current_date - timedelta(days=28)).tz_localize(None)
    fetch_data_to = (current_date + timedelta(hours=1)).tz_localize(None)
    
    feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)
    ts_data = feature_view.get_batch_data(
        start_time=fetch_data_from,
        end_time=fetch_data_to
    )

    # Sort and process data without filtering
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)
    
    # Create feature matrix keeping all hours
    location_ids = ts_data['pickup_location_id'].unique()
    x = np.ndarray(shape=(len(location_ids), config.N_FEATURES), dtype=np.float32)
    
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i, :] = ts_data_i['rides'].values[:config.N_FEATURES]

    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(config.N_FEATURES))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    
    return features
# load_batch_of_features_from_store(current_date)
    
# %%
