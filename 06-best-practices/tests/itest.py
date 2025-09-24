import pandas as pd
from datetime import datetime
import os
import boto3
import logging
from tabulate import tabulate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define helper function to create datetime objects
def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

# Function to get the S3 client
def get_s3_client():
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")  # Localstack default endpoint
    return boto3.client("s3", endpoint_url=s3_endpoint_url)

# Create the test data and save it to S3
def save_test_data_to_s3():
    """
    Creates a test dataframe, saves it locally, and uploads it to Localstack S3.
    """
    logger.info("Creating test DataFrame for January 2023...")
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    # Displaying input DataFrame
    logger.info("Displaying the DataFrame:")
    print(tabulate(df_input, headers="keys", tablefmt="grid", showindex=False))

    # Save to S3
    save_data(df_input, "nyc-duration", "in/2023-01.parquet")

def save_data(df, bucket_name, key):
    """
    Save a DataFrame to a given S3 bucket and key as a Parquet file.
    """
    s3_client = get_s3_client()

    # Save DataFrame locally
    local_file_path = "temp_data.parquet"
    df.to_parquet(
        local_file_path,
        engine='pyarrow',
        compression=None,
        index=False
    )
    
    # Upload the local file to S3
    try:
        logger.info(f"Uploading DataFrame to S3 at {bucket_name}/{key}...")
        s3_client.upload_file(local_file_path, bucket_name, key)
        logger.info(f"Successfully uploaded {key} to bucket {bucket_name}.")
    except Exception as e:
        logger.error("Failed to upload to S3.", exc_info=e)
        raise
    finally:
        # Clean up the local temp file
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

def run_batch_job():
    """
    Runs the batch.py script to process the input data and save output to S3.
    """
    logger.info("Running batch.py to process data...")
    exit_code = os.system("python latest_batch.py 2023 1")
    if exit_code != 0:
        logger.error("batch.py failed to execute correctly.")
        raise RuntimeError("Batch job failed!")

def read_results_and_verify():
    """
    Reads the output file from S3, verifies the data, and computes the sum of predicted durations.
    """
    logger.info("Reading the output file from S3...")
    s3_client = get_s3_client()
    output_file = "s3://nyc-duration/out/2023-01.parquet"

    # Use s3fs to read the Parquet file
    df_output = pd.read_parquet(output_file, storage_options={
        'client_kwargs': {
            'endpoint_url': os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
        }
    })
 #   logger.info("Displaying the output DataFrame:")
 #   print(tabulate(df_output, headers="keys", tablefmt="grid", showindex=False))

    # Compute the sum of predicted durations
    predicted_duration_sum = df_output["predicted_duration"].sum()
    logger.info(f"Sum of predicted durations: {predicted_duration_sum}")
    return predicted_duration_sum

def main():
    """
    Main entry point for the integration test.
    """
    logger.info("Starting the integration test...")

    try:
        # Step 1: Save test data to S3
        save_test_data_to_s3()

        # Step 2: Run the batch processing job
        run_batch_job()

        # Step 3: Read and verify the results
        result = read_results_and_verify()
        logger.info(f"The sum of predicted durations is: {result}")

    except Exception as e:
        logger.error("Integration test failed.", exc_info=e)


if __name__ == "__main__":
    main()
