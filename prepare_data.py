

#%%
import numpy as np
import os
import pickle
import pandas as pd
import yaml
import re
# from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher


columns_dtypes  = {
    'DOLocationID': 'int64',
    'PULocationID':'int64',
    'RatecodeID': 'float64',
    'VendorID': 'int64',
    'congestion_surcharge': 'float64',
    'extra': 'float64',
    'fare_amount': 'float64',
    'improvement_surcharge': 'float64',
    'mta_tax': 'float64',
    'passenger_count': 'float64',
    'payment_type': 'float64',
    'store_and_fwd_flag':  'O',
    'tip_amount': 'float64',
    'tolls_amount': 'float64',
    'total_amount': 'float64',
    'trip_distance': 'float64'
    }



#%%

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


# def apply_column_types(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
#     # Load YAML config
#     with open(config_path) as f:
#         config = yaml.safe_load(f)
    
#     for pattern, dtype in config.get("columns", {}).items():
#         # Find all columns matching this regex
#         matching_cols = [c for c in df.columns if re.match(pattern, c)]
        
#         for col in matching_cols:
#             # Apply the correct dtype
#             if dtype == "datetime":
#                 df[col] = pd.to_datetime(df[col], errors="coerce")
#             else:
#                 df[col] = df[col].astype(dtype, errors="ignore")
    
#     return df

def apply_column_types(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    
    # Load YAML config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    for var_name, dtype in config.get("columns", {}).items():
        variables =  [c for c in df.columns if re.match(var_name, c)]
        for var in variables:
            # Apply the correct dtype
            if df[var].dtype != dtype:
                print(f"expected dtype: {dtype}; actual dtype: {df[var].dtype}")
                df[var] = df[var].astype(dtype, errors="ignore")
            if dtype == "datetime":
                print(f"the column {var} is type {df[var].dtype}")
                df[var] = pd.to_datetime(df[var], errors="coerce")
                print(f"column {var} converted to {df[var].dtype}")
            # else:
            #     df[col] = df[col].astype(dtype, errors="ignore")
    
    return df




def _download_tlc_file(filename,
                    base_url="https://d37ci6vzurychx.cloudfront.net/trip-data",
                    output_dir = None,
                    saving = False,
                    # processing = True
                    ):

    # Generates file_name creates df from file and downloads if saving is True. 
    # filename assumed to be in the correct format '{cartype}_tripdata_{YYYY}-{MM}.parquet' 
    file_path = ('/').join([base_url, filename])
    df = pd.read_parquet(file_path)
    if saving:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            dump_pickle(df, os.path.join(output_dir, filename))
        else:
            print("Saving directory not provided, file not saved")
    return df

def _processing(df,  numeric_col_names = [], categorical_cols_names = [], training = False):
   

    
    assert numeric_col_names or categorical_cols_names, "At least a list of numeric columns or a list of categorical columns must be given"
    
    #--1. Keep only list of categorical and numerical values
    columns_to_keep = numeric_col_names if numeric_col_names else categorical_cols_names
    columns_to_keep.extend(categorical_cols_names)
    print(numeric_col_names)
    print(categorical_cols_names)
    print(columns_to_keep)
    df = df[columns_to_keep]

    #--2. Restrict `trip_distance` column between limits: 
    #     for yellow  taxis between 0 and 20 miles
    if 'trip_distance' in df.columns:
        limit = df.trip_distance.describe(percentiles=[0.99])
        df = df[(df.trip_distance >= 1.0) & (df.trip_distance < limit )]
   
    #--3. Create feature duration
    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime'] 

    #--4. Create combined feature pickup-droppoff
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)

     #--5. Encode categorical variables with DicVectorizer or
    # fit_transf = True if training else False
    # X, dv = _encoding(df, dv, columns_to_encode = None, fit_transf=fit_transf)
    # return X,dv
    return df


def _encoding(df: pd.DataFrame, dv, columns_to_encode = None, 
                    fit_transf = False):

    # Assums columns have already been dropped
    # Checks that the columns of the dataframe are correct (optional???)
    # (if there are extracoulumn)


    if columns_to_encode is not None:
        df_temp = df.copy()[columns_to_encode]
    else:
        df_temp = df.copy()
    dicts = df_temp.to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


# def encoding_orig (df: pd.DataFrame, dv: DictVectorizer,
#                 fit_dv: bool = False):
#     df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
#     categorical = ['PU_DO']
#     numerical = ['trip_distance']
#     dicts = df[categorical + numerical].to_dict(orient='records')
#     if fit_dv:
#         X = dv.fit_transform(dicts)
#     else:
#         X = dv.transform(dicts)
#     return X, dv

def prepare_data():
    #--1 preprocess data  
    #--2 split data in training and test  
    
    #--3 encode training data --> update DictVectorzer of FeatureHasher (fit_trnsform = True)
    dv = DictVectorizer()
    dv = FeatureHasher(n_features = 2*df.shape[1])
    X_train, dv = _encoding(df_train, dv, columns_to_encode = None, fit_transf=True)
    
    #--4 preprocess test data (fit_trnsform = False)
    X_train, dv = _encoding(df_train, dv, columns_to_encode = None, fit_transf=True)
    pass

#%%

def download_tlc_data(start_date, end_date,
                      output_dir='data',
                      car_type='yellow',
                      base_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data"):
    """
    Downloads NYC TLC trip record data for a specified date range.

    Args:
        start_date (str): The start date in 'YYYY-MM' format.
        end_date (str): The end date in 'YYYY-MM' format.
        output_dir (str): The directory to save the downloaded files.
        car_type (str): The type of taxi ('yellow', 'green', 'fhv').
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate a date range for each month
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # generates names of files to download
    for date in date_range:
        year_month = date.strftime('%Y-%m')  
        file_name = base_url+ f"//{car_type}_tripdata_year_month.parquet"
        processed_file_name = f"{car_type}_tripdata_year_month_processed.parquet"
    # check if file processed file is already stored
        if  processed_file_name in os.listdir(output_dir):
            df = pd.read_parquet(os.path.join(output_dir, processed_file_name))
        else:
            df = _download_tlc_file(date, 
                                car_type = car_type, 
                                base_url=base_url,
                                processing = True)
        

    



#%%
if __name__ == "__main__":
    filename = "/home/simona/Documents/Portfolio/NYTraffic/data/green_tripdata_2021-01.parquet"
    config_path = "/home/simona/Documents/Portfolio/NYTraffic/datatypes.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    dftest = pd.read_parquet(filename)
    # numeric_columns = config.get("numeric_columns") if "numeric_columns" in config.keys() else []
    # categorical_columns = config.get("categorical_columns") if "categorical_columns" in config.keys() else []
 
    
    dftest_2 = apply_column_types(dftest, config_path)
    
    #-- PREPROCESSING
    columns_to_keep = config.get("columns_to_keep")
    if columns_to_keep:
        columns_to_keep = [col for col in columns_to_keep if col in dftest_2.columns]
        dftest_2 = dftest_2[columns_to_keep].copy()

    # dftest_2['duration'] = dftest_2['lpep_dropoff_datetime'] - dftest_2['lpep_pickup_datetime'] 
    # dftest_2['duration'] = dftest_2['duration'].apply(lambda x:x.total_seconds()/60)


    # limit_duartion = dftest_2.duration.describe(percentiles=[0.99])['99%']
    limit_dist = dftest_2.trip_distance.describe(percentiles=[0.99])['99%']
    dftest_2 = dftest_2[(dftest_2.trip_distance >= 1.0) & (dftest_2.trip_distance < limit_dist )]

    dftest_2['PU_DO'] = dftest_2['PULocationID'].astype(str) + '_' + dftest_2['DOLocationID'].astype(str)

    # -- ENCODING


# %%
categorical_columns = [c for c in dftest_2.columns if dftest_2[c].dtype=='O']
column_to_encode=['PU_DO']
training = True
# encoder_params = {'n_features':32}
# encoder = FeatureHasher(**encoder_params)

hasher = FeatureHasher(n_features=32, input_type='string')  # You can set n_features as needed
hashed_features = hasher.transform([[val] for val in dftest_2['PU_DO'].astype(str)])
# Convert hashed features to a DataFrame
hashed_df = pd.DataFrame(hashed_features.toarray(), 
                         columns=[f'PU_DO_hash_{i}' for i in range(hashed_features.shape[1])],
                         index=dftest_2.index)

# Concatenate hashed features with the original DataFrame (drop PU_DO if you don't want the original)
encoded_df = pd.concat([dftest_2.drop(columns=['PU_DO']), hashed_df], axis=1)

encoded_df.head()



# %%
