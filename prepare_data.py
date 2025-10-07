

#%%
import numpy as np
import os
import pickle
import pandas as pd
import yaml
import re
# from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.preprocessing import OneHotEncoder



#%%

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)



def apply_column_types(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    for var_name, dtype in config.get("columns", {}).items():
        variables =  [c for c in df.columns if re.match(var_name, c)]
        for var in variables:
            if df[var].dtype != dtype:
                print(f"expected dtype: {dtype}; actual dtype: {df[var].dtype}")
                df[var] = df[var].astype(dtype, errors="ignore")
            if dtype == "datetime":
                print(f"the column {var} is type {df[var].dtype}")
                df[var] = pd.to_datetime(df[var], errors="coerce")
                print(f"column {var} converted to {df[var].dtype}")
    
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




def _encoding_original(df: pd.DataFrame, dv, columns_to_encode = None, 
                    fit_transf = False):

    # Assumes columns have already been dropped
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
def _preprocessing(df, selected_features = None):
    

    dist_cutoff = df.trip_distance.describe(percentiles=[0.99])['99%']
    df = df[(df.trip_distance >= 1.0) & (df.trip_distance < dist_cutoff )].copy()

    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    df.drop(['PULocationID','DOLocationID'],axis=1, inplace=True)
    PU_time_column = [c for c in df.columns if 'pickup_datetime' in c][0]
    DO_time_column = [c for c in df.columns if 'dropoff_datetime' in c][0]
    df['duration']  = df[DO_time_column]-df[PU_time_column]
    df['duration'] = df['duration'].apply(lambda x:x.total_seconds()/60)
    if selected_features:
        selected_features = [col for col in selected_features if col in df.columns]
        selected_features.extend(['PU_DO', 'duration'])
        df = df[selected_features]

    return df

def _encoding(df, n_features):

    encoder = FeatureHasher(n_features=n_features, input_type='string')  # You can set n_features as needed
    data = [[val] for val in df['PU_DO'].astype(str)] # Featurehasher expects lists
    encoded_features = encoder.transform(data)
   
    # Convert hashed features to a DataFrame
    hashed_df = pd.DataFrame(encoded_features.toarray(), 
                            columns=[f'PU_DO_hash_{i}' for i in range(encoded_features.shape[1])],
                            index=df.index)

    # Concatenate hashed features with the original DataFrame (drop PU_DO if you don't want the original)
    encoded_df = pd.concat([df.drop(columns=['PU_DO']), hashed_df], axis=1)
    return encoded_df, encoder

def encode_and_concat(df, columns, encoder, drop_original=True, training = False):
    """
    Applies a scikit-learn encoder to specified columns and returns the DataFrame
    with encoded features appended.
    """
    X = df[columns]
    encoder_name = encoder.__class__.__name__

    if encoder_name == "FeatureHasher":
        if len(columns) == 1:
            encoded = encoder.transform(X[columns[0]].astype(str).apply(lambda x: [x]))
        else:
            encoded = encoder.transform(X.astype(str).to_dict(orient="records"))
        encoded_df = pd.DataFrame(encoded.toarray(),
                                  columns=[f"{'_'.join(columns)}_hash_{i}" for i in range(encoded.shape[1])],
                                  index=df.index)
    elif encoder_name == "DictVectorizer":
        # Convert DataFrame to list of dicts
        encoded = encoder.fit_transform(X.astype(str).to_dict(orient="records"))
        try:
            feature_names = encoder.get_feature_names_out()
        except Exception:
            feature_names = [f"{col}_dv_{i}" for i in range(encoded.shape[1])]
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
    else:
        try:
            encoded_arr = encoder.transform(X)
        except AttributeError:
            encoded_arr = encoder.fit_transform(X)
        if hasattr(encoded_arr, "toarray"):  # sparse
            encoded_arr = encoded_arr.toarray()
        try:
            col_names = encoder.get_feature_names_out(columns)
        except Exception:
            col_names = [f"{col}_enc_{i}" for col in columns for i in range(encoded_arr.shape[1] // len(columns))]
        encoded_df = pd.DataFrame(encoded_arr, columns=col_names, index=df.index)
    
    if drop_original:
        df_out = pd.concat([df.drop(columns=columns), encoded_df], axis=1)
    else:
        df_out = pd.concat([df, encoded_df], axis=1)
    
    return df_out
#%%


#--1. Load data from file: 
    # TO DO: check if the data required is stored, if not download from link


#--2.Preprocessing:
#       -removing rows with `trip-distance in theleft or right tails
#       -Combininig columns `PULocationID` and  `DOLocationID` and dropping original columns
#       -Creating column `duration` (which is the target variable)
#       -Dropping all the unnecessary columns
     

#--3 split data in training and test and  validation, based on dates entered if given, 
#    otherwise last 20% of dates (YYY-MM-DD) as test data last 10% validation data.
# 
#--4 Encoding: 
#       -Encode train data with fit_transform if applicable. 
#       -Encode test and validation 
# 
# -- 5 Saving    
#       -Create dest_path folder unless it already exists 
#       -Save encoder and datasets

_encoding
#%%



#%%



if __name__ == "__main__":
    filename = "/home/simona/Documents/Portfolio/NYTraffic/data/green_tripdata_2021-01.parquet"
    config_path = "/home/simona/Documents/Portfolio/NYTraffic/datatypes.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    dftest = pd.read_parquet(filename) 
    
    dftest_2 = apply_column_types(dftest, config_path)
    
    #-- PREPROCESSING
    columns_to_keep = config.get("columns_to_keep")
    dftest_2 = _preprocessing(dftest_2, selected_features = columns_to_keep)

    #--SPLITTING
  

    # -- ENCODING
    dftest_2, _ =_encoding(dftest_2, n_features=32)



# %%

#-- ENCODING GENERALIZED USING `encode_and_concat`

# FeatureHasher example
hasher = FeatureHasher(n_features=16, input_type='string')
df_hashed = encode_and_concat(df, ['PU_DO'], hasher)

# DictVectorizer example
dv = DictVectorizer(sparse=False)
df_dv = encode_and_concat(df, ['PU_DO'], dv)

# OneHotEncoder example
ohe = OneHotEncoder(sparse=False)
df_ohe = encode_and_concat(df, ['PU_DO'], ohe)

# %%
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
dataset = 'green'
train_start = '2021-01-01'; train_end = '2021-07-31'
train_start_date = datetime.strptime(train_start, '%Y-%m-%d')
train_end_date = datetime.strptime(train_end, '%Y-%m-%d')
if train_end_date.strftime('%Y-%m')  == train_start_date.strftime('%Y-%m'):
    train_start_file= f"{dataset}_tripdata_{train_start_date.strftime('%Y-%m')}"
else:
    next_date = train_start_date.replace(day=1) 
    dates_list =[]
    while next_date <= train_end_date:
        dates_list.append(next_date.strftime('%Y-%m'))
        next_date += relativedelta(months = 1) 
    files_list = [f"{dataset}_tripdata_{date}" for date in dates_list]


# print(f"file ")

# print(train_end_date - train_start_date)

# print(train_start_date - timedelta(days=10)) 

#%%

from dateutil.relativedelta import relativedelta

train_start_date = datetime.strptime('2020-11-06', '%Y-%m-%d')
train_end_date = datetime.strptime('2021-02-03', '%Y-%m-%d')

