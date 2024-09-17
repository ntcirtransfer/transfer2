FROM python:3.8-slim-buster

RUN pip3 install --no-cache-dir numpy torch torchvision scikit-learn pandas
RUN python -c "from torchvision import models; models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)"

WORKDIR /app

COPY models models
COPY predict.py .
COPY dpr_models.py .
COPY submission.sh .

# docker build -t my-dmr-runs .
# docker run --rm -v "/home/yamamoto/DMR/:/app/" my-dmr-runs bash ./submission.sh input/input_file.csv output/output_file.csv Lifelog-6
# docker save my-dmr-runs | gzip > my-dmr-runs.tar.gz
# docker load < my-dmr-runs.tar.gz