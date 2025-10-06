# Use Debian slim as the base image
FROM python:3.13-slim-bullseye

# Install system dependencies required for building Python packages and Kafka dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    cmake \
    pkg-config \
    libssl-dev \
    libsasl2-dev \
    librdkafka-dev \
    librdkafka1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
    
# We are actually working with confluent_Kafka version 2.6.1. 
# RUN pip install --no-cache-dir \
# confluent_Kafka

# Python 3.13 requires this to be compatible with pytorch
RUN pip install --upgrade typing_extensions

ARG CACHE_BUST=1

# Clone the repository into the 'wandber' folder
RUN git clone https://github.com/DIETI-DISTA-IoT/Wandber.git wandber

# Set the working directory inside the container
WORKDIR /wandber

# Install the dependencies specified in the requirements file and other required libraries
RUN pip install --no-cache-dir -r requirements.txt

# Command to start the data simulator script when the container is run
CMD ["python", "wandber.py"]
