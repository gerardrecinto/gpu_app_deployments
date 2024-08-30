FROM nvidia/cuda:11.4-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3-pip && \
    pip3 install torch torchvision

# Copy your PyTorch script into the container
COPY train.py /workspace/

WORKDIR /workspace/

# Command to run the PyTorch script
CMD ["python3", "train.py"]

