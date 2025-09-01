# Custom L1 Distance layer module

# Import dependencies
import torch
from torch import nn

# Siamese L1 Distance Class
class L1Dist(nn.Module):

    # Init method - Inheritence
    def __init__(self):
        super().__init__()

    # Similarity Calculation
    def forward(self, input_embedding, validation_embedding):
        return torch.abs(input_embedding - validation_embedding)


class EmbeddingModel(nn.Module):
    def __init__(self, input_shape: int):
        super().__init__()

        # First Block
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Second Block
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Third Block
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Fourth Block
        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4),
            nn.ReLU()
        )

        # Classifier Block
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=9216, out_features=4096),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor):
        return self.classifier(self.block_4(self.block_3(self.block_2(self.block_1(x)))))


class SiameseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingModel(input_shape=3)
        self.Dist = L1Dist()
        self.classifier = nn.Linear(in_features=4096, out_features=1)

    def forward(self, anchor_image, validation_image):
        return self.classifier(self.Dist(self.embedding(anchor_image), self.embedding(validation_image)))

