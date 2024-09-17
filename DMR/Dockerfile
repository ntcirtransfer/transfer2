FROM python:3.8-slim-buster

RUN pip3 install --no-cache-dir numpy torch torchvision scikit-learn pandas
RUN python -c "from torchvision import models; models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)"

WORKDIR /app

COPY models models
COPY predict.py .
COPY dpr_models.py .
COPY submission.sh .