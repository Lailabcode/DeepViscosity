# Use an official Python 3.9 image as base
FROM python:3.9-slim

# Install system dependencies and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        ca-certificates \
        autoconf \
        automake \
        libtool \
        make \
        && rm -rf /var/lib/apt/lists/*

# Download, build, and install HMMER 3.3.2
RUN wget http://eddylab.org/software/hmmer/hmmer-3.3.2.tar.gz && \
    tar -xzf hmmer-3.3.2.tar.gz && \
    cd hmmer-3.3.2 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf hmmer-3.3.2 hmmer-3.3.2.tar.gz

# Install ANARCI from source (for CLI usage)
RUN git clone https://github.com/oxpig/ANARCI.git /opt/ANARCI && \
    cd /opt/ANARCI && \
    pip install .
ENV PATH="/opt/ANARCI/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir keras==2.11.0 tensorflow-cpu==2.11.0 scikit-learn==1.0.2 pandas numpy==1.26.4 joblib dill biopython

# Copy the project files into the container
WORKDIR /app
COPY . /app

# Set the entrypoint to run the main script
ENTRYPOINT ["python", "deepviscosity_predictor.py"]
