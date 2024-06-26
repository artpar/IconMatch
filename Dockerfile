FROM python:3.8

# Install required system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install numpy --prefer-binary
RUN pip install .
ENTRYPOINT python icondetection/demo/server.py