FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install build dependencies
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
RUN pip3 install --no-cache-dir torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install all AudioLDM dependencies
RUN pip3 install --no-cache-dir \
    tqdm \
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
    diffusers \
    gradio==3.22.1

# Install AudioLDM
RUN pip3 install --no-cache-dir git+https://github.com/haoheliu/AudioLDM.git

# Clone only the necessary files from repository
RUN git clone --depth 1 https://github.com/haoheliu/AudioLDM . && \
    rm -rf .git

# Clean up pip cache and unnecessary files
RUN find /usr/local/lib/python3.8/dist-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.8/dist-packages -name "__pycache__" -delete && \
    rm -rf /root/.cache/pip

# Create a smaller final image
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/python3.8 /usr/bin/python3

# Set working directory
WORKDIR /app

# Copy installed Python packages and application files from builder stage
COPY --from=builder /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages
COPY --from=builder /usr/local/bin/audioldm /usr/local/bin/audioldm
COPY --from=builder /app /app

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
