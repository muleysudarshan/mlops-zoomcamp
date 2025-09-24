#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os

def read_data(filename):
    """
    Reads input data from a Parquet file.

    Parameters:
    - filename: path to the input file
    
    Returns:
    - Raw DataFrame
    """
    return pd.read_parquet(filename)

def prepare_data(df, categorical):
    """
    Prepares data by applying transformations.

    Parameters:
    - df: Input DataFrame
    - categorical: List of categorical columns to preprocess
    
    Returns:
    - Transformed DataFrame
    """
    # Calculate trip duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    # Filter out records with invalid durations
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Handle missing values and convert categorical columns to strings
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def main(year, month):
    """
    Main function that processes data for a given year and month, and saves the results.

    Parameters:
    - year: Year for the input data (int)
    - month: Month for the input data (int)
    """
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_dir = 'output'
    output_file = f'{output_dir}/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    # Ensure the output directory exists (creates if it doesn't)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    # Define categorical columns
    categorical = ['PULocationID', 'DOLocationID']

    # Read input data
    df = read_data(input_file)
    
    # Process the dataframe
    df = prepare_data(df, categorical)

    # Create a unique ride ID
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Transform categorical features and predict
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    # Save results
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)
    print('Results save to:', {output_file})

if __name__ == "__main__":
    """
    Entry point of the script
    """
    # Get arguments from the command line
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    # Call the main function
    main(year, month)
