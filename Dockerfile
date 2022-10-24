FROM nvcr.io/nvidia/tensorrt:22.09-py3
RUN apt update -y \
    && apt install python3.8 -y \
    && apt install wget -y \
    && apt install python3.8-distutils -y \
    && rm -rf /var/lib/apt/lists/*
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py
COPY . /app/
RUN ls -la /app/*
WORKDIR /app
RUN pip3.8 install --no-cache ./
RUN pip3.8 install --no-cache onnxruntime-gpu
