# Base image from RunPod
FROM runpod/base:0.4.0-cuda11.8.0

# Set a working directory
WORKDIR /app

# --- Optional: System dependencies ---
# COPY builder/setup.sh /setup.sh
# RUN /bin/bash /setup.sh && \
#     rm /setup.sh

# Install Python dependencies
COPY builder/requirements.txt /app/requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --no-cache-dir -r /app/requirements.txt

# Add the source files
COPY src /app/src

# Create a directory for saving the model
RUN mkdir -p /app/models
VOLUME ["/app/models"]

# Set the entry point to run the Python script
CMD ["python3.11", "-u", "/app/src/app.py"]
