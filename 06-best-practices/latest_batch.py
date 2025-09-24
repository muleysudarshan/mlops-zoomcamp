#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os
import boto3
from botocore.exceptions import ClientError

def get_input_path(year, month):
    """
    Generate the input file path dynamically based on environment variable or default URL.

    Returns:
        str: Input file path.
    """
    default_input_pattern = 's3://nyc-duration/in/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    """
    Generate the output file path dynamically based on environment variable or default pattern.

    Returns:
        str: Output file path.
    """
    default_output_pattern = 's3://nyc-duration/out/{year:04d}-{month:02d}-predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def ensure_bucket_exists(bucket_name):
    """
    Ensure that an S3 bucket exists in Localstack.

    Parameters:
        bucket_name (str): Name of the bucket to ensure exists.
    """
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    s3 = boto3.client('s3', endpoint_url=s3_endpoint_url)

    try:
        response = s3.head_bucket(Bucket=bucket_name)
        print(f"[INFO] Bucket '{bucket_name}' exists and is accessible.")
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"[INFO] Bucket '{bucket_name}' does not exist. Creating it now...")
            s3.create_bucket(Bucket=bucket_name)
            print(f"[INFO] Bucket '{bucket_name}' created successfully.")
        else:
            print(f"[ERROR] Error accessing bucket '{bucket_name}': {e}")
            raise


def read_data(filename):
    """
    Reads input data from a Parquet file.

    Parameters:
        filename (str): File path (local or S3 URL).

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')  # Localstack endpoint

    # Configure storage options for Localstack S3
    storage_options = {
        'client_kwargs': {
            'endpoint_url': s3_endpoint_url
        }
    }

    print(f"[DEBUG] Reading data from: {filename}")
    print(f"[DEBUG] Storage options: {storage_options}")

    try:
        # Read the parquet file using pandas
        df = pd.read_parquet(filename, storage_options=storage_options)
    except Exception as e:
        print(f"[ERROR] Error while reading data: {e}")
        raise

    print("[INFO] Data successfully read.")
    print(f"[DEBUG] DataFrame head:\n{df.head()}")
    return df


def prepare_data(df, categorical):
    """
    Prepares data by transforming duration and categorical features.

    Returns:
        pandas.DataFrame: Processed data.
    """
    print("[INFO] Preparing data...")
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    # Filter out invalid durations
    print("[DEBUG] Filtering invalid durations...")
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Preprocess categorical columns
    print("[DEBUG] Filling missing values and converting categorical columns to strings...")
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    print("[INFO] Data preparation complete.")
    print(f"[DEBUG] Processed DataFrame head:\n{df.head()}")
    return df


def main(year, month):
    """
    Main function that processes data and saves results.

    Parameters:
        year (int): Year for the input data.
        month (int): Month for the input data.
    """
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    ensure_bucket_exists("nyc-duration")

    # Load the ML model
    print("[INFO] Loading the model...")
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    print("[INFO] Model loaded successfully.")

    # Define categorical columns
    categorical = ['PULocationID', 'DOLocationID']

    print(f"[INFO] Reading input data for {year}-{month}...")
    df = read_data(input_file)

    print("[INFO] Processing input data...")
    df = prepare_data(df, categorical)

    print("[INFO] Making predictions...")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(f"[INFO] Mean predicted duration: {y_pred.mean()}")

    print(f"[INFO] Saving results to {output_file}...")
    df_result = pd.DataFrame({
        'ride_id': f'{year:04d}/{month:02d}_' + df.index.astype('str'),
        'predicted_duration': y_pred,
    })
    storage_options = {
        "client_kwargs": {
            "endpoint_url": os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
        }
    }
    df_result.to_parquet(output_file, engine="pyarrow", index=False, storage_options=storage_options)
    print("[INFO] Results saved successfully.")


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
