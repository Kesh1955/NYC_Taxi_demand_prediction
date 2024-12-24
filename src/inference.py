from datetime import datetime, timedelta

import hopsworks
# from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

import src.config as config
from src.feature_store_api import get_feature_store, get_or_create_feature_view
from src.config import FEATURE_VIEW_METADATA

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """"""
    # past_rides_columns = [c for c in features.columns if c.startswith('rides_')]
    predictions = model.predict(features)

    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)
    
    return results

def diagnose_timeseries_completeness(ts_data, n_features):
    """
    Diagnoses why the time series data might be incomplete by checking various aspects
    of the data structure and identifying potential gaps.
    
    Args:
        ts_data (pd.DataFrame): The time series data to analyze
        n_features (int): Expected number of features per location
    """
    # 1. Basic data overview
    print("=== Data Overview ===")
    print(f"Total records: {len(ts_data)}")
    print(f"Unique locations: {len(ts_data['pickup_location_id'].unique())}")
    print(f"Expected features per location: {n_features}")
    print(f"Expected total records: {n_features * len(ts_data['pickup_location_id'].unique())}")
    
    # 2. Check for missing hours in each location
    print("\n=== Time Series Continuity Analysis ===")
    for location_id in ts_data['pickup_location_id'].unique():
        location_data = ts_data[ts_data.pickup_location_id == location_id]
        n_records = len(location_data)
        
        if n_records != n_features:
            print(f"\nLocation ID {location_id} has {n_records} records (expected {n_features})")
            
            # Show the time range for this location
            print(f"Time range: {location_data['pickup_hour'].min()} to {location_data['pickup_hour'].max()}")
            
            # Find missing hours
            hours = pd.date_range(
                start=location_data['pickup_hour'].min(),
                end=location_data['pickup_hour'].max(),
                freq='H'
            )
            missing_hours = set(hours) - set(location_data['pickup_hour'])
            if missing_hours:
                print(f"Missing {len(missing_hours)} hours:")
                print(sorted(missing_hours)[:5], "..." if len(missing_hours) > 5 else "")
    
    # 3. Check time range consistency
    print("\n=== Time Range Analysis ===")
    print(f"Overall time range: {ts_data['pickup_hour'].min()} to {ts_data['pickup_hour'].max()}")
    total_hours = len(pd.date_range(
        start=ts_data['pickup_hour'].min(),
        end=ts_data['pickup_hour'].max(),
        freq='H'
    ))
    print(f"Total hours in range: {total_hours}")

    return {
        'total_records': len(ts_data),
        'unique_locations': len(ts_data['pickup_location_id'].unique()),
        'expected_records': n_features * len(ts_data['pickup_location_id'].unique()),
        'is_complete': len(ts_data) == n_features * len(ts_data['pickup_location_id'].unique())
    }

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
        
        # Debugging output to identify the issue
        rides_values = ts_data_i['rides'].values
        #print(f"Location ID: {location_id}")
        #print(f"Number of data points: {len(rides_values)}")
        #print(f"Expected: {config.N_FEATURES}")
        
        if len(rides_values) < config.N_FEATURES:
            print(f"** Insufficient data for location_id {location_id}. Skipping or handling required. **")
            continue  # Skip this iteration for now
        
        if len(rides_values) > config.N_FEATURES:
            print(f"** More data than expected for location_id {location_id}. Truncating to fit. **")
        
        # Assignment (will trigger an error if len(rides_values) < config.N_FEATURES)
        x[i, :] = rides_values[:config.N_FEATURES]

    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(config.N_FEATURES))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids

    return features


# def load_batch_of_features_from_store(
#     current_date: pd.Timestamp,    
# ) -> pd.DataFrame:
#     """Fetches the batch of features used by the ML system at `current_date`

#     Args:
#         current_date (datetime): datetime of the prediction for which we want
#         to get the batch of features

#     Returns:
#         pd.DataFrame: 4 columns:
#             - `pickup_hour`
#             - `rides`
#             - `pickup_location_id`
#             - `pickpu_ts`
#     """
#     n_features = config.N_FEATURES


#     feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)
    

#     # fetch data from the feature store
#     # current_date = pd.to_datetime('2024-10-31 00:00:00')
#     current_date = pd.to_datetime('2024-10-30 23:00:00')
#     fetch_data_from = current_date - timedelta(days=28)
#     fetch_data_to = current_date - timedelta(hours=1)

#     # add plus minus margin to make sure we do not drop any observation
#     ts_data = feature_view.get_batch_data(
#         start_time=fetch_data_from - timedelta(days=1),
#         end_time=fetch_data_to + timedelta(days=1)
#     )
    
#     # filter data to the time period we are interested in
#     pickup_ts_from = int(fetch_data_from.timestamp() * 1000)
#     pickup_ts_to = int(fetch_data_to.timestamp() * 1000)
#     ts_data = ts_data[ts_data.pickup_ts.between(pickup_ts_from, pickup_ts_to)]

#     # sort data by location and time
#     ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)

#     # validate we are not missing data in the feature store
#     location_ids = ts_data['pickup_location_id'].unique()

#     # Add this before the assertion
#     diagnostics = diagnose_timeseries_completeness(ts_data, n_features)

#     assert len(ts_data) == config.N_FEATURES * len(location_ids), \
#         "Time-series data is not complete. Make sure your feature pipeline is up and runnning."

#     # transpose time-series data as a feature vector, for each `pickup_location_id`
#     x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
#     for i, location_id in enumerate(location_ids):
#         ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
#         ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
#         x[i, :] = ts_data_i['rides'].values

#     # numpy arrays to Pandas dataframes
#     features = pd.DataFrame(
#         x,
#         columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
#     )
#     features['pickup_hour'] = current_date
#     features['pickup_location_id'] = location_ids
#     features.sort_values(by=['pickup_location_id'], inplace=True)

#     return features
    

def load_model_from_registry():
    
    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )  
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir)  / 'model.pkl')
       
    return model

def load_predictions_from_store(
    from_pickup_hour: datetime,
    to_pickup_hour: datetime
    ) -> pd.DataFrame:
    
    from src.config import FEATURE_VIEW_PREDICTIONS_METADATA
    from src.feature_store_api import get_or_create_feature_view

    predictions_fv = get_or_create_feature_view(FEATURE_VIEW_PREDICTIONS_METADATA)
    print(f"Feature View Object: {predictions_fv}")

    # Add timezone info logging
    print(f"Input time range:")
    print(f"From: {from_pickup_hour} (timezone: {from_pickup_hour.tzinfo})")
    print(f"To: {to_pickup_hour} (timezone: {to_pickup_hour.tzinfo})")
    
    # Note we're fetching with a wider window
    fetch_start = from_pickup_hour - timedelta(days=1)
    fetch_end = to_pickup_hour + timedelta(days=1)
    print(f"Fetching window:")
    print(f"From: {fetch_start}")
    print(f"To: {fetch_end}")

    predictions = predictions_fv.get_batch_data(
        start_time=fetch_start,
        end_time=fetch_end
    )
    
    print(f"Raw predictions shape: {predictions.shape}")
    print(f"Raw predictions time range: {predictions['pickup_hour'].min()} to {predictions['pickup_hour'].max()}")

    # UTC conversion
    # After getting predictions from feature store
    predictions['pickup_hour'] = pd.to_datetime(predictions['pickup_hour']).dt.tz_localize(None)
    from_pickup_hour = pd.to_datetime(from_pickup_hour).tz_localize(None)
    to_pickup_hour = pd.to_datetime(to_pickup_hour).tz_localize(None)

    # print("Available columns:", predictions.columns)
    # print("\nPredictions dataframe head:")
    # print(predictions.head())

    # Filter to desired range
    predictions = predictions[predictions.pickup_hour.between(from_pickup_hour, to_pickup_hour)]
    print(f"Filtered predictions shape: {predictions.shape}")
    print(f"Filtered time range: {predictions['pickup_hour'].min()} to {predictions['pickup_hour'].max()}")

    predictions.sort_values(by=['pickup_hour', 'pickup_location_id'], inplace=True)

    # Before returning, let's check what columns we have
    print("Columns in predictions before returning:")
    print(predictions.columns)
    
    # Also check a sample of the data
    print("\nSample of predictions data:")
    print(predictions.head())
    return predictions

# # %%
# from src.feature_store_api import get_or_create_feature_view
# feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)
# # %%
