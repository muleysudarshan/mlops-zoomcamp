#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
from prefect import flow, task
import argparse

# Task 1: Data Preprocessing
@task
def preprocess_data(file_path: str, categorical_features: list):
    """
    Reads and preprocesses the parquet file for scoring.
    - Calculates trip duration.
    - Removes outliers.
    - Cleans missing data.
    """
    # Read the Parquet file
    df = pd.read_parquet(file_path)

    # Calculate trip durations in minutes
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60

    # Filter trips with valid durations only
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()

    # Handle missing values in categorical features
    df[categorical_features] = df[categorical_features].fillna(-1).astype('int').astype('str')

    return df

# Task 2: Load Model
@task
def load_model(model_file: str):
    """
    Loads the trained model and DictVectorizer from a pickle file.
    """
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)

    print("Model and DictVectorizer loaded successfully.")
    return dv, model

# Task 3: Score Data
@task
def score_data(df, dv, model, categorical_features: list):
    """
    Scores the given dataset using the provided dictionary vectorizer and model.
    - Calculates standard deviation and mean of predictions.
    """
    # Transform categorical features
    dicts = df[categorical_features].to_dict(orient='records')
    X_val = dv.transform(dicts)

    # Make predictions
    y_pred = model.predict(X_val)

    # Calculate statistics
    std_dev = np.std(y_pred)
    mean_pred = np.mean(y_pred)

    print(f"Mean predicted duration: {mean_pred:.2f}")

    return std_dev, mean_pred

# Flow: Orchestrating the tasks
@flow
def batch_inference_flow(year: int, month: int, model_file: str, categorical_features: list):
    """
    Prefect flow for batch inference.
    """
    # Path to input dataset
    test_file_path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    print(f"Fetching data from: {test_file_path}")

    # Step 1: Preprocess Data
    df = preprocess_data(test_file_path, categorical_features)

    # Step 2: Load Model
    dv, model = load_model(model_file)

    # Step 3: Score Data
    std_dev, mean_pred = score_data(df, dv, model, categorical_features)

    # Final Output
    print("\nBatch Inference Completed.")
    print(f"Final Results - Mean Prediction: {mean_pred:.2f}")

    return std_dev, mean_pred

if __name__ == "__main__":
    # Command Line Argument Parsing
    parser = argparse.ArgumentParser(description="Batch inference script using Prefect")
    parser.add_argument('--year', type=int, default='2023', required=True, help='Year of the dataset')
    parser.add_argument('--month', type=int, default='5', required=True, help='Month of the dataset')
    parser.add_argument('--model_file', type=str, default='model.bin', help='Path to the model file')
    args = parser.parse_args()

    # Categorical features used for scoring
    categorical_features = ['PULocationID', 'DOLocationID']

    # Execute the Prefect Flow
    batch_inference_flow(
        year=args.year,
        month=args.month,
        model_file=args.model_file,
        categorical_features=categorical_features
    )
