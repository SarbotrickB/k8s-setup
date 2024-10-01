FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

# Copy application code
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Run the application
CMD ["python3", "app.py"]
