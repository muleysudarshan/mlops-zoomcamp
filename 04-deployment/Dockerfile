FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory inside the container
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir prefect pandas pyarrow scikit-learn==1.5.0

# Copy your Python script to the working directory
#COPY duration_prediction_docker.py .
COPY duration_prediction_prefect_docker.py .

# Set default command to invoke the script with CLI arguments passed by the user
#ENTRYPOINT ["python", "duration_prediction_docker.py"]
ENTRYPOINT ["python", "duration_prediction_prefect_docker.py"]
