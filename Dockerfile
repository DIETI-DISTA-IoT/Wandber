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

# Full rebuild bust: pass CACHE_BUST=<timestamp> to re-run pip install AND code clone.
# Used by:  make build-wandber-scache
ARG CACHE_BUST=1

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Python 3.13 requires this to be compatible with pytorch
RUN pip install --upgrade typing_extensions

# Install dependencies from the build context (submodule checkout on disk).
# This layer is cached when using scache-nolib; re-run only when using scache.
COPY wandber/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Code-only bust: pass CODE_BUST=<timestamp> to re-run only the git clones, keeping pip cached.
# Used by:  make build-wandber-scache-nolib
ARG CODE_BUST=1

RUN git clone --branch sereBench https://github.com/DIETI-DISTA-IoT/Wandber.git /wandber

WORKDIR /wandber

RUN git clone --branch sereBench https://github.com/DIETI-DISTA-IoT/of-core OpenFAIR/

CMD ["python", "manager_server.py"]
