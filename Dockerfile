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

# Create a wrapper script to handle output directory
RUN echo '#!/bin/bash\n\
# Check if output directory is mounted\n\
if [ -d "/app/output" ]; then\n\
    # Run the analysis\n\
    python deepviscosity_predictor.py\n\
    \n\
    # Copy output files to mounted output directory\n\
    if [ -f "DeepViscosity_classes.csv" ]; then\n\
        cp DeepViscosity_classes.csv /app/output/\n\
    fi\n\
    if [ -f "DeepSP_descriptors.csv" ]; then\n\
        cp DeepSP_descriptors.csv /app/output/\n\
    fi\n\
    echo "Analysis completed. Output files copied to mounted directory."\n\
else\n\
    # Run without output directory mounting\n\
    python deepviscosity_predictor.py\n\
    echo "Analysis completed. Output files in current directory."\n\
fi' > /app/run_analysis.sh && chmod +x /app/run_analysis.sh

# Set the entrypoint to run the wrapper script
ENTRYPOINT ["/app/run_analysis.sh"]
