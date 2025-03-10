FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and clean up in the same layer
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    git \
    ffmpeg \
    libsndfile1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/python3.8 /usr/bin/python3

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install AudioLDM separately to avoid conflicts with specific PyTorch version
RUN pip3 install --no-cache-dir numpy==1.23.5 \
    && pip3 install --no-cache-dir --no-deps git+https://github.com/haoheliu/AudioLDM.git \
    && pip3 install --no-cache-dir transformers==4.30.2 diffusers==0.19.3 \
    && find /usr/local/lib/python3.8/dist-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.8/dist-packages -name "__pycache__" -delete \
    && rm -rf /root/.cache/pip

# Set the entrypoint
ENTRYPOINT ["audioldm"]

# Default command (can be overridden)
CMD ["--help"]
