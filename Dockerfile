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

# Install PyTorch with CUDA support (specific version required by AudioLDM)
RUN pip3 install --no-cache-dir torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install all AudioLDM dependencies as per the module's requirements
RUN pip3 install --no-cache-dir \
    tqdm \
    gradio==3.22.1 \
    pyyaml \
    einops \
    chardet \
    numpy==1.23.5 \
    soundfile \
    librosa==0.9.2 \
    scipy \
    pandas \
    torchlibrosa==0.0.9 \
    transformers==4.29.0 \
    progressbar \
    ftfy \
    diffusers

# Install AudioLDM
RUN pip3 install --no-cache-dir git+https://github.com/haoheliu/AudioLDM.git

# Additional cleanup
RUN find /usr/local/lib/python3.8/dist-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.8/dist-packages -name "__pycache__" -delete \
    && rm -rf /root/.cache/pip

# Clone the repository to get the app.py file
RUN git clone https://github.com/haoheliu/AudioLDM .

# Make the entrypoint script
COPY <<EOF /app/entrypoint.sh
#!/bin/bash
if [ "\$1" = "webapp" ]; then
    # Run the Gradio web app
    python app.py
else
    # Run audioldm with arguments
    audioldm "\$@"
fi
EOF

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD ["--help"]
