

#%%
import numpy as np
import os
import pickle
import pandas as pd
import yaml
import re
# from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher



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

def _encoding(df, n_features, training = False):

    encoder = FeatureHasher(n_features=n_features, input_type='string')  # You can set n_features as needed
    data = [[val] for val in df['PU_DO'].astype(str)] # Featurehasher expects lists
    if training:
        encoded_features = encoder.fit_transform(data)
    else:
        encoded_features = encoder.transform(data)
    # Convert hashed features to a DataFrame
    hashed_df = pd.DataFrame(encoded_features.toarray(), 
                            columns=[f'PU_DO_hash_{i}' for i in range(encoded_features.shape[1])],
                            index=df.index)

    # Concatenate hashed features with the original DataFrame (drop PU_DO if you don't want the original)
    encoded_df = pd.concat([df.drop(columns=['PU_DO']), hashed_df], axis=1)
    return encoded_df, encoder


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

    # -- ENCODING
    dftest_2 =_encoding(dftest_2, n_features=32)

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
