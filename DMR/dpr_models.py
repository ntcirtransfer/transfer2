import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# Define the image encoder using a pre-trained ResNet18 model
class ImageEncoder(nn.Module):
    def __init__(self, output_dim, lock_pretrained_model=True):
        super(ImageEncoder, self).__init__()
        self.model_imagenet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model_imagenet.heads = nn.Identity()  # Remove the classification head

        self.model_places = models.__dict__['resnet18'](num_classes=365)
        checkpoint = torch.load('./models/resnet18_places365.pth.tar', map_location=lambda storage, loc: storage, weights_only=False)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model_places.load_state_dict(state_dict)
        self.model_places.heads = nn.Identity() # Remove the classification head

        self.fc = nn.Linear(1365, output_dim)
        
        # Do not update parameters of pre-trained models
        if lock_pretrained_model:
            for param in self.model_imagenet.parameters():
                param.requires_grad = False
            for param in self.model_places.parameters():
                param.requires_grad = False
                    
    def forward(self, x):
        imagenet_h = self.model_imagenet(x)
        places_h = self.model_places(x)
        h = torch.concat([imagenet_h, places_h], axis=1)
        return self.fc(h)

# Define the sensor data encoder using a simple feed-forward network
class SensorEncoder(nn.Module):
    def __init__(self, avg, std, input_dim, hidden_dim, output_dim, normalization=True):
        super(SensorEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.avg = avg
        self.std = std
        self.normalization = normalization

    def forward(self, x):
        if self.normalization:
            x = (x - self.avg) / self.std
        return self.fc(x)
    
# Define the DPR model
class DPRModel(nn.Module):
    def __init__(self, image_encoder, sensor_encoder):
        super(DPRModel, self).__init__()
        self.image_encoder = image_encoder
        self.sensor_encoder = sensor_encoder

    def forward(self, image_data, sensor_data):
        image_embs = self.image_encoder(image_data)
        sensor_embs = self.sensor_encoder(sensor_data)
        return image_embs, sensor_embs

# Define the similarity based corss entropy loss
class SimilarityBasedCrossEntropy(nn.Module):
    def __init__(self, temperature=0.1, device='cpu'):
        super(SimilarityBasedCrossEntropy, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, image_emb, sensor_emb):
        # Calculate similarity matrix (mini-batch size x mini-batch size)
        sim_matrix = torch.cosine_similarity(
            image_emb.unsqueeze(1),
            sensor_emb.unsqueeze(0),
            dim=2,
        )

        # Calculate similarity based cross entropy
        target = torch.arange(sim_matrix.shape[0]).to(self.device)
        loss = nn.functional.cross_entropy(sim_matrix / self.temperature, target)

        return loss