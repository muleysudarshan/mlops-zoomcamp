import os
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def save_data(df, filename):
    """
    Saves a DataFrame to a given S3 path in Localstack.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The path to save the file (e.g., S3 path).
    """
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    options = {
        'client_kwargs': {
            'endpoint_url': s3_endpoint_url
        }
    }

    print(f"Saving data to: {filename}")
    try:
        df.to_parquet(
            filename,
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options
        )
        print("Data successfully saved.")
    except Exception as e:
        print(f"Error while saving data: {e}")
        raise


def create_test_data():
    """
    Creates the test dataframe.

    Returns:
        pd.DataFrame: The test dataframe.
    """
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    return pd.DataFrame(data, columns=columns)


def read_results(filename):
    """
    Reads the results file from Localstack S3.

    Returns:
        pd.DataFrame: The result dataframe.
    """
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    options = {
        'client_kwargs': {
            'endpoint_url': s3_endpoint_url
        }
    }

    print(f"Reading results from: {filename}")
    try:
        df = pd.read_parquet(filename, storage_options=options)
        print("Results successfully read.")
        return df
    except Exception as e:
        print(f"Error while reading results: {e}")
        raise


def main():
    """
    Integration test for processing and saving ride predictions via batch.py.
    """
    # Step 1: Create test data and save it to Localstack S3
    df_input = create_test_data()
    input_file = "s3://nyc-duration/in/2023-01.parquet"
    save_data(df_input, input_file)

    # Step 2: Run the batch process for January 2023
    print("Running batch.py script...")
    os.system("python latest_batch.py 2023 1")  # Run batch.py for January 2023

    # Step 3: Read back results and verify
    output_file = "s3://nyc-duration/out/2023-01.parquet"
    df_result = read_results(output_file)

    # Step 4: Compute sum of predicted durations
    predicted_sum = df_result['predicted_duration'].sum()
    print(f"Sum of predicted durations: {predicted_sum}")


if __name__ == "__main__":
    main()
