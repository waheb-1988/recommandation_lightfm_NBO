FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    pkg-config \
    python3-dev \
    git \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    cd /tmp && \
    git clone https://github.com/lyst/lightfm.git && \
    cd lightfm && \
    git checkout 1.17 && \
    sed -i "s/__builtins__.__LIGHTFM_SETUP__ = True/import builtins; builtins.__LIGHTFM_SETUP__ = True/" setup.py && \
    pip install . && \
    cd / && \
    rm -rf /tmp/lightfm

# Copy app code
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
