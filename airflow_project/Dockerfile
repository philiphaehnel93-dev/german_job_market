# Use a modern Python 3.11 environment to get pre-compiled wheels
FROM apache/airflow:2.8.1-python3.11

# Temporarily switch to root to install system-level build tools
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the restricted airflow user
USER airflow

COPY requirements.txt /requirements.txt

# Upgrade pip and install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0.0" && \
    pip install --no-cache-dir -r /requirements.txt

# Download the spaCy model
RUN python -m spacy download de_core_news_sm
