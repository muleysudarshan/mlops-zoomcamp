#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pickle
import os
import argparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def preprocess_data(file_path, categorical_features):
    # Read the file
    df = pd.read_parquet(file_path)

    # Calculate trip duration in minutes
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60

    # Filter trips with invalid durations
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()

    # Handle missing values for categorical columns
    df[categorical_features] = df[categorical_features].fillna(-1).astype('int').astype('str')

    return df

# def train_model(file_path, categorical_features):
#     """
#     Trains a Linear Regression model using given features and target duration.
#     Saves the model and DictVectorizer as model.bin for later use.
#     """
#     # Preprocess training data
#     df = preprocess_data(file_path, categorical_features)

#     # Transform categorical features to dictionary format
#     train_dicts = df[categorical_features].to_dict(orient='records')

#     # Vectorize categorical features
#     dv = DictVectorizer()
#     X_train = dv.fit_transform(train_dicts)

#     # Target variable: trip duration
#     y_train = df['duration'].values

#     # Train a Linear Regression model
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Evaluate the model on the training set
#     y_pred = model.predict(X_train)
#     rmse = mean_squared_error(y_train, y_pred)
#     print(f"Training RMSE: {rmse:.2f}")

#     # Save the model and DictVectorizer to a file
#     with open("model.bin", "wb") as f_out:
#         pickle.dump((dv, model), f_out)
#     print("Model and vectorizer saved successfully!")

#     return rmse

def score_model(file_path, categorical_features):
    """
    Scores the given model on new data and calculates standard deviation of predictions.
    """
    # Load the trained model and DictVectorizer
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # Preprocess the data
    df = preprocess_data(file_path, categorical_features)

    # Transform the data using DictVectorizer
    dicts = df[categorical_features].to_dict(orient='records')
    X_val = dv.transform(dicts)

    # Predict durations
    y_pred = model.predict(X_val)

    # Standard deviation of predictions
    std_dev = np.std(y_pred)
    print(f"Standard deviation of the predicted durations: {std_dev:.2f}")

    mean_pred = np.mean(y_pred)
    print(f"Mean predicted duration: {mean_pred:.2f}")

    return std_dev, mean_pred

# def prepare_and_save_results(file_path, categorical_features, year, month, output_file):
#     """
#     Prepares the DataFrame with ride_ids and predicted durations, 
#     then saves the output to a Parquet file.
#     """
#     # Preprocess the March 2023 data
#     df = preprocess_data(file_path, categorical_features)

#     # Load the trained model and DictVectorizer
#     with open('model.bin', 'rb') as f_in:
#         dv, model = pickle.load(f_in)
        
#     # Create the ride_id column based on year, month, and ride index
#     df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

#     # Transform features and predict durations
#     dicts = df[categorical_features].to_dict(orient='records')
#     X_val = dv.transform(dicts)
#     y_pred = model.predict(X_val)

#     # Create the result DataFrame
#     df_result = pd.DataFrame({
#         'ride_id': df['ride_id'],
#         'predicted_duration': y_pred
#     })

#     # Save the results to a Parquet file
#     df_result.to_parquet(
#         output_file,
#         engine='pyarrow',
#         compression=None,
#         index=False
#     )
#     print(f"Results saved to {output_file}")

#     # Calculate and print the size of the output file
#     output_file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert bytes to MB
#     print(f"Size of the output file: {output_file_size:.2f} MB")

#     return output_file_size

if __name__ == "__main__":

    # CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Script for predicting taxi ride durations.")
    parser.add_argument('--year', type=int, default='2023', required=True, help='Year of the dataset')
    parser.add_argument('--month', type=int, default='3', required=True, help='Month of the dataset')
    parser.add_argument('--output_file_path', type=str, default='output.parquet', help='Path to save the output Parquet file')

    args = parser.parse_args()

    # Paths to datasets
    train_file_path = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet'
    test_file_path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year:04d}-{args.month:02d}.parquet'

    # Categorical features
    categorical_features = ['PULocationID', 'DOLocationID']

    # year = 2023
    # month = 3
    # output_file_path = "output.parquet" 

    # print("\nStep 1: Training the Model")
    # # Train the model and save it
    # train_rmse = train_model(train_file_path, categorical_features)

    print("\nStep 2: Scoring the Model")
    # Score the model on the March 2023 dataset and calculate standard deviation
    test_std_dev, test_mean_pred = score_model(test_file_path, categorical_features)

    # print("\nStep 3: Preparing and Saving Results")
    # # Prepare the results and save to a Parquet file
    # output_file_size = prepare_and_save_results(
    #     test_file_path, categorical_features, args.year, args.month, args.output_file_path
    # )
