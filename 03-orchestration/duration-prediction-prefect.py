from prefect import flow, task
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --------------------------------
# Phase 1: Data Loading & Preparation
# --------------------------------

# Task: Read the raw data
@task
def read_data(filename: str) -> pd.DataFrame:
    """Task to read raw Parquet data."""
    df = pd.read_parquet(filename)
    return df

# Task: Prepare the data
@task
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Task to prepare the data by filtering durations and converting categorical columns."""
    # Calculate trip duration in minutes
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60

    # Filter durations between 1 and 60 minutes
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    
    # Convert selected columns to categorical (string type)
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

# Task: Count records in a DataFrame
@task
def count_records(df: pd.DataFrame, step_name: str) -> int:
    """Task to count and print the number of records in a DataFrame."""
    num_records = len(df)
    print(f"Number of records after {step_name}: {num_records:,}")
    return num_records

# Task: Log metrics to MLflow for data processing
@task
def log_to_mlflow(step_name: str, file_path: str, num_records: int):
    """Task to log parameters and metrics to MLflow."""
    mlflow.set_experiment("nyc_taxi_experiment")  # Set or create MLflow experiment
    
    with mlflow.start_run():
        # Log data-specific parameters and metrics
        mlflow.log_param("file_path", file_path)
        mlflow.log_param("step_name", step_name)  # Step-related log (e.g. 'raw', 'prepared')
        mlflow.log_metric("num_records", num_records)

        print(f"Logged step={step_name}, file_path={file_path}, num_records={num_records} to MLflow.")

# --------------------------------
# Phase 2: Model Training & Registration
# --------------------------------

# Task: Train a Linear Regression model
@task
def train_model(df: pd.DataFrame):
    """Task to train a Linear Regression model using a DictVectorizer."""
    # Features: pickup and dropoff locations
    categorical = ['PULocationID', 'DOLocationID']
    
    # Transform dataset to dictionary format
    train_dicts = df[categorical].to_dict(orient='records')
    
    # Fit DictVectorizer
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    
    # Target variable: trip durations
    y_train = df['duration'].values
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print intercept of the model
    intercept = model.intercept_
    print(f"Model intercept: {intercept:.2f}")
    
    # Evaluate performance on training data
    y_pred = model.predict(X_train)
    rmse = mean_squared_error(y_train, y_pred)
    print(f"Training RMSE: {rmse:.2f}")
    
    return dv, model, intercept, rmse

# Task: Log model, metrics, and register model to MLflow
@task
def log_and_register_model(dv: DictVectorizer, model: LinearRegression, intercept: float, rmse: float):
    """Logs a trained model, DictVectorizer, and metrics to MLflow and registers the model."""
    mlflow.set_experiment("nyc_taxi_experiment")  # Set the MLflow experiment

    with mlflow.start_run() as run:  # Start an MLflow run
        # Log model training parameters and metrics
        mlflow.log_param("model_type", "Linear Regression")
        mlflow.log_param("num_features", len(dv.feature_names_))
        mlflow.log_metric("intercept", intercept)
        mlflow.log_metric("rmse", rmse)
        
        # Log the trained model
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("Model logged to MLflow.")
        
        # Register the model in the MLflow Model Registry
        model_name = "nyc_taxi_linear_model"
        registered_model = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
        print(f"Model registered with name: {model_name}, version: {registered_model.version}")

# --------------------------------
# Prefect Flow
# --------------------------------

@flow
def taxi_pipeline():
    """Pipeline for processing raw and filtered taxi data, training a model, and registering with MLflow."""
    # Dataset file path
    file_path = './data/yellow_tripdata_2023-03.parquet'
    
    # Phase 1: Load and prepare data
    raw_df = read_data(file_path)
    raw_count = count_records(raw_df, step_name="raw data")
    log_to_mlflow(step_name="raw data", file_path=file_path, num_records=raw_count)
    
    prepared_df = prepare_data(raw_df)
    prepared_count = count_records(prepared_df, step_name="prepared data")
    log_to_mlflow(step_name="prepared data", file_path=file_path, num_records=prepared_count)
    
    # Phase 2: Train model and log to MLflow
    dv, model, intercept, rmse = train_model(prepared_df)
    log_and_register_model(dv, model, intercept, rmse)

if __name__ == "__main__":
    taxi_pipeline()
