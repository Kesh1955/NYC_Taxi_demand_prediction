from datetime import datetime, timedelta
from argparse import ArgumentParser

import pandas as pd

import src.config as config
from src.logger import get_logger
from src.config import FEATURE_GROUP_PREDICTIONS_METADATA, FEATURE_GROUP_METADATA
from src.feature_store_api import get_or_create_feature_group, get_feature_store

logger = get_logger()


def load_predictions_and_actual_values_from_store(
    from_date: datetime,
    to_date: datetime,
) -> pd.DataFrame:
    """Fetches model predictions and actuals values from
    `from_date` to `to_date` from the Feature Store and returns a dataframe

    Args:
        from_date (datetime): min datetime for which we want predictions and
        actual values

        to_date (datetime): max datetime for which we want predictions and
        actual values

    Returns:
        pd.DataFrame: 4 columns
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
            - `rides`
    """
    # 2 feature groups we need to merge
    # predictions_fg = get_or_create_feature_group(FEATURE_GROUP_PREDICTIONS_METADATA) 
    # actuals_fg = get_or_create_feature_group(FEATURE_GROUP_METADATA)

    import hopsworks
    import src.config as config
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY)

    fs = project.get_feature_store()
    predictions_fg = fs.get_feature_group(name ='model_predictions_feature_group', version =4)
    actuals_fg = fs.get_feature_group(name ='time_series_hourly_feature_group', version = 3)

    # query to join the 2 features groups by `pickup_hour` and `pickup_location_id`
    from_ts = int(from_date.timestamp() * 1000)
    to_ts = int(to_date.timestamp() * 1000)
    query = predictions_fg.select_all() \
        .join(actuals_fg.select(['pickup_location_id', 'pickup_ts', 'rides']),
              on=['pickup_ts', 'pickup_location_id'], prefix=None) \
        .filter(predictions_fg.pickup_ts >= from_ts) \
        .filter(predictions_fg.pickup_ts <= to_ts)
    
    # breakpoint()

    # create the feature view `config.FEATURE_VIEW_MONITORING` if it does not
    # exist yet
    feature_store = get_feature_store()
    try:
        # create feature view as it does not exist yet
        feature_store.create_feature_view(
            name=config.MONITORING_FV_NAME,
            version=6,
            query=query
        )
    except:
        logger.info('Feature view already existed. Skip creation.')

    # feature view
    monitoring_fv = feature_store.get_feature_view(
        name=config.MONITORING_FV_NAME,
        version=6
    )
    
    # fetch data form the feature view
    # fetch predicted and actual values for the last 30 days
    monitoring_df = monitoring_fv.get_batch_data(
        start_time=from_date - timedelta(days=7),
        end_time=to_date + timedelta(days=7),
    )

    # filter data to the time period we are interested in
    pickup_ts_from = int(from_date.timestamp() * 1000)
    pickup_ts_to = int(to_date.timestamp() * 1000)
    monitoring_df = monitoring_df[monitoring_df.pickup_ts.between(pickup_ts_from, pickup_ts_to)]

    return monitoring_df

if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--from_date',
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
                        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
    parser.add_argument('--to_date',
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
                        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()


    monitoring_df = load_predictions_and_actual_values_from_store()









##########################################################################
# logger = get_logger()

# def load_predictions_and_actual_values_from_store(
#     from_date: datetime,
#     to_date: datetime,
# ) -> pd.DataFrame:
#     """Fetches model predictions and actuals values from
#     `from_date` to `to_date` from the Feature Store and returns a dataframe

#     Args:
#         from_date (datetime): min datetime for which we want predictions and
#         actual values

#         to_date (datetime): max datetime for which we want predictions and
#         actual values

#     Returns:
#         pd.DataFrame: 4 columns
#             - `pickup_location_id`
#             - `predicted_demand`
#             - `pickup_hour`
#             - `rides`
#     """
#     # 2 feature groups we need to merge
#     predictions_fg = get_or_create_feature_group(FEATURE_GROUP_PREDICTIONS_METADATA)
#     print(predictions_fg.schema)
#     predictions_df = predictions_fg.read()
#     print(predictions_df.head(5))


#     actuals_fg = get_or_create_feature_group(FEATURE_GROUP_METADATA)
#     print(actuals_fg.schema)
#     actuals_df = actuals_fg.read()
#     print(actuals_df.head(5))

    
#     # query to join the 2 features groups by `pickup_hour` and `pickup_location_id`
#     from_ts = int(from_date.timestamp() * 1000)
#     to_ts = int(to_date.timestamp() * 1000)


#     # First, let's try to get the SQL query
#     try:
#         query = predictions_fg.select_all() \
#             .join(
#                 actuals_fg.select(['pickup_location_id', 'pickup_ts', 'rides']),
#                 on=['pickup_ts', 'pickup_location_id' ],
#                 prefix=None,
#             ) \
#             .filter(predictions_fg.pickup_ts >= from_ts) \
#             .filter(predictions_fg.pickup_ts <= to_ts)

#         # Try different methods to get SQL - one of these might work
#         try:
#             print("Method 1 - _get_query():")
#             print(query._get_query())
#         except:
#             pass
            
#         try:
#             print("\nMethod 2 - get_query_string():")
#             print(query.get_query_string())
#         except:
#             pass
        
#         try:
#             print("\nMethod 3 - _sql():")
#             print(query._sql())
#         except:
#             pass

#     except Exception as e:
#         print("Failed to construct query:", str(e))

#     # # add column with Unix epoch milliseconds
#     # from_ts  = from_date.astype(int) // 10**6

# # Let's first try reading each feature group separately to ensure they work
#     # try:
#     #     predictions_query = predictions_fg.select_all() \
#     #         .filter(predictions_fg.pickup_ts >= from_ts) \
#     #         .filter(predictions_fg.pickup_ts <= to_ts)
#     #     predictions_data = predictions_query.read()
#     #     print("Predictions query successful")
#     # except Exception as e:
#     #     print("Predictions query failed:", str(e))

#     # try:
#     #     actuals_query = actuals_fg.select(['pickup_location_id', 'pickup_ts', 'rides'])
#     #     actuals_data = actuals_query.read()
#     #     print("Actuals query successful")
#     # except Exception as e:
#     #     print("Actuals query failed:", str(e))

#     # try:
#     #     # Let's first check what the feature group metadata shows
#     #     print("Predictions FG Schema:", predictions_fg.schema)
#     #     print("Actuals FG Schema:", actuals_fg.schema)
        
#     #     # query to join the 2 features groups by `pickup_hour` and `pickup_location_id`


#     #     query = predictions_fg.select_all() \
#     #         .join(
#     #             actuals_fg.select(['pickup_location_id', 
#     #                             actuals_fg.pickup_ts.cast('bigint').alias('pickup_ts'),  # Explicitly cast to bigint 
#     #                             'rides']),
#     #             on=['pickup_location_id'],  # Remove pickup_ts from the join condition
#     #             prefix='actuals'
#     #         ) \
#     #         .asof_join('pickup_ts', 'actualspickup_ts') \
#     #         .filter(predictions_fg.pickup_ts.cast('bigint') >= from_ts) \
#     #         .filter(predictions_fg.pickup_ts.cast('bigint') <= to_ts)
                
#     #     joined_data = query.read()
#     #     print("Join successful")
#     # except Exception as e:
#     #     print("Join failed:", str(e))
    
# # Can you also share what version of Hopsworks you're using? 
# # This might help us understand if this is a known issue

        

#     # query = predictions_fg.select_all() \
#     # .join(
#     #     actuals_fg.select(['pickup_location_id', 'pickup_ts', 'rides']),
#     #     on=['pickup_location_id', 'pickup_ts'],  # Include pickup_ts in the join condition
#     #     prefix='actuals'
#     # ) \
#     # .filter(predictions_fg.pickup_ts >= from_ts) \
#     # .filter(predictions_fg.pickup_ts <= to_ts)

    
#     # create the feature view `config.FEATURE_VIEW_MONITORING` if it does not
#     # exist yet
#     feature_store = get_feature_store()
#     try:
#         # create feature view as it does not exist yet
#         feature_store.create_feature_view(
#             name=config.MONITORING_FV_NAME,
#             version=5, # config.MONITORING_FV_VERSION
#             query=query
#         )
#     except:
#         logger.info('Feature view already existed. Skip creation.')

#     # feature view
#     monitoring_fv = feature_store.get_feature_view(
#         name=config.MONITORING_FV_NAME,
#         version=5
#     )
    
#     # fetch data form the feature view
#     # fetch predicted and actual values for the last 30 days
#     monitoring_df = monitoring_fv.get_batch_data(
#         start_time=from_date - timedelta(days=7),
#         end_time=to_date + timedelta(days=7),
#     )

#     # filter data to the time period we are interested in
#     pickup_ts_from = int(from_date.timestamp() * 1000)
#     pickup_ts_to = int(to_date.timestamp() * 1000)
#     monitoring_df = monitoring_df[monitoring_df.pickup_ts.between(pickup_ts_from, pickup_ts_to)]

#     return monitoring_df


# # def load_predictions_and_actual_values_from_store(
# #     from_date: datetime,
# #     to_date: datetime,
# # ) -> pd.DataFrame:
# #     """Fetches model predictions and actuals values from
# #     `from_date` to `to_date` from the Feature Store and returns a dataframe

# #     Args:
# #         from_date (datetime): min datetime for which we want predictions and
# #         actual values

# #         to_date (datetime): max datetime for which we want predictions and
# #         actual values

# #     Returns:
# #         pd.DataFrame: 4 columns
# #             - `pickup_location_id`
# #             - `predicted_demand`
# #             - `pickup_hour`
# #             - `rides`
# #     """
# #     # 2 feature groups we need to merge
# #     predictions_fg = get_or_create_feature_group(FEATURE_GROUP_PREDICTIONS_METADATA)
    
# #     from_ts = pd.to_datetime(from_date).tz_localize(None)
# #     to_ts = pd.to_datetime(to_date).tz_localize(None)
# #     # from_ts = int(from_date.timestamp() * 1000)
# #     print('from_ts : ',from_ts)
# #     # to_ts = int(to_date.timestamp() * 1000)   
# #     print('to ts : ', to_ts)

# #     # predictions_fg = predictions_fg.get_batch_data(
# #     #     start_time=from_ts,
# #     #     end_time=to_ts
# #     # )

# #     #print(f"Raw predictions shape: {predictions_fg.shape}")
# #     #print(f"Raw predictions time range: {predictions_fg['pickup_hour'].min()} to {predictions_fg['pickup_hour'].max()}")

    
# #     actuals_fg = get_or_create_feature_group(FEATURE_GROUP_METADATA)

# #     # query to join the 2 features groups by `pickup_hour` and `pickup_location_id`
    
# #     query = predictions_fg.select_all() \
# #         .join(actuals_fg.select(['pickup_location_id', 'pickup_ts', 'rides']),
# #               on=['pickup_ts', 'pickup_location_id'], prefix='actuals')

# #     # query = predictions_fg.select_all() \
# #     #     .join(actuals_fg.select(['pickup_location_id', 'pickup_hour', 'rides']),
# #     #           on=['pickup_hour', 'pickup_location_id'], 
# #     #           prefix='actuals') \
# #     #     .filter(predictions_fg.get_feature('pickup_hour') >= from_ts) \
# #     #     .filter(predictions_fg.get_feature('pickup_hour') <= to_ts)

# #     # query = predictions_fg.select_all() \
# #     #     .join(actuals_fg.select(['pickup_location_id', 'pickup_hour', 'rides']),
# #     #           on=['pickup_hour', 'pickup_location_id'], prefix='actuals'
# #     #           ) \
# #     #     .filter(predictions_fg.pickup_hour >= from_ts) \
# #     #     .filter(predictions_fg.pickup_hour <= to_ts)
    
# #     # breakpoint()

# #     # create the feature view `config.FEATURE_VIEW_MONITORING` if it does not
# #     # exist yet
# #     feature_store = get_feature_store()

# #     print("Starting feature view retrieval/creation process")

# #     try:
# #         print("Attempting to get existing feature view...")
# #         monitoring_fv = feature_store.get_feature_view(
# #             name=config.MONITORING_FV_NAME,
# #             version=config.MONITORING_FV_VERSION
# #         )
# #         print(f"Successfully retrieved existing feature view: {config.MONITORING_FV_NAME}")
        
# #         # Verify the query is valid
# #         print("Query being used:", query)

# #         current_date = pd.to_datetime('2024-10-30 23:00:00')
        
# #         # Try to read some data to verify the feature view is working
# #         try:
# #             test_data = monitoring_fv.get_batch_data(
# #                 start_time= current_date - pd.Timedelta(days=1),
# #                 end_time= current_date
# #             )
# #             print("Successfully read test data from feature view")
# #         except Exception as read_error:
# #             print(f"Error reading from feature view: {str(read_error)}")

# #     except Exception as e:
# #         print(f"Failed to get feature view with error: {str(e)}")

# #     # feature view
# #     monitoring_fv = feature_store.get_feature_view(
# #         name=config.MONITORING_FV_NAME,
# #         version=config.MONITORING_FV_VERSION
# #     )
    
# #     # fetch data form the feature view
# #     # fetch predicted and actual values for the last 30 days
# #     monitoring_df = monitoring_fv.get_batch_data(
# #         start_time=from_date - timedelta(days=7),
# #         end_time=to_date + timedelta(days=7),
# #     )

# #     # filter data to the time period we are interested in
# #     pickup_ts_from = int(from_date.timestamp() * 1000)
# #     pickup_ts_to = int(to_date.timestamp() * 1000)
# #     monitoring_df = monitoring_df[monitoring_df.pickup_ts.between(pickup_ts_from, pickup_ts_to)]

# #     return monitoring_df

# if __name__ == '__main__':

#     # parse command line arguments
#     parser = ArgumentParser()
#     parser.add_argument('--from_date',
#                         type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
#                         help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
#     parser.add_argument('--to_date',
#                         type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
#                         help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
#     args = parser.parse_args()


#     monitoring_df = load_predictions_and_actual_values_from_store()

#     # %%
# # import hopsworks
# # import src.config as config
# # project = hopsworks.login(
# #         project=config.HOPSWORKS_PROJECT_NAME,
# #         api_key_value=config.HOPSWORKS_API_KEY
# #     )

# # fs = project.get_feature_store()
# # # # %%
# # actuals_fg = fs.get_feature_group(name ='time_series_hourly_feature_group', version = 3)

# # print(actuals_fg.schema)
# # # # %%
# # actuals_df = actuals_fg.read()
# # actuals_df.head(5)
# # # # %%
# # predictions_fg = fs.get_feature_group(name ='model_predictions_feature_group', version =1)
# # print(predictions_fg.schema)
 
# # # # %%
# # predictions_df =predictions_fg.read()
# # predictions_df.head(5)
# # # %%
# # actuals_df.tail()
# # # %%

# # # %%
# # comparison_df = predictions_df.merge(
# #     actuals_df,
# #     on=['pickup_hour', 'pickup_location_id'],
# #     how='left',
# #     suffixes=('_pred', '_actual')
# # )
# # comparison_df

# # # # %% 
# # # # Add comparison metrics
# # # comparison_df['prediction_error'] = comparison_df['predicted_demand'] - comparison_df['rides']

# # # # View aligned predictions and actuals
# # # print(comparison_df[['pickup_hour', 'pickup_location_id', 'predicted_demand', 'rides', 'prediction_error']])
# # # # %%

# # # import src.config as config
# # # from src.logger import get_logger
# # # from src.config import FEATURE_GROUP_PREDICTIONS_METADATA, FEATURE_GROUP_METADATA
# # # from src.feature_store_api import get_or_create_feature_group, get_feature_store
# # # actuals_fg = get_or_create_feature_group(FEATURE_GROUP_METADATA)
# # # actuals_fg.features
# # # # %% 
# # # predictions_fg = get_or_create_feature_group(FEATURE_GROUP_PREDICTIONS_METADATA)
# # # a = predictions_fg.read()
# # # a.columns
# # # # %%
# # # feature_store = get_feature_store()
# # # print(feature_store.get_feature_groups(name = 'model_predictions_feature_group'))
# # # # %%
# # # # Get feature groups list
# # # fg_list = feature_store.get_feature_groups(name = 'model_predictions_feature_group')
# # # # Get first feature group
# # # test_fg = fg_list[0]
# # # # Print its name and schema
# # # print(f"Name: {test_fg.name}")
# # # print(f"Schema: {test_fg.schema}")

# # # df_test = test_fg.read()
# # # df_test.columns
# # # %%
# # %% 
# # print(predictions_fg.schema)
# # print(actuals_fg.schema)

# # %%
