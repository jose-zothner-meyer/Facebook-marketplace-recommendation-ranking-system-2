import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class FineTunedResNet(nn.Module):
    """
    Modified ResNet-50 model for transfer learning.
    
    Key changes:
      - Directly replaces the original fc layer with a sequential block (via self.new_layers) instead of creating a separate combined model.
      - This avoids applying AdaptiveAvgPool2d twice.
      - Only layer4 and fc layers (inside the original model) are unfrozen.
    """
    def __init__(self, num_classes: int) -> None:
        super(FineTunedResNet, self).__init__()
        
        # Load pre-trained ResNet-50 using the new weights syntax.
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all layers initially.
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze parameters only in layer4 and the fc layer.
        for name, layer in self.model.named_children():
            if name in ['layer4', 'fc']:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Replace the fc layer with a sequential block.
        # According to the teacher’s design, we first apply a ReLU activation,
        # then use a linear layer to map the 1000 features to the desired number of classes.
        self.new_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, num_classes)  # Final layer outputs class scores.
        )
        # Combine the base model and the new classification head.
        self.combined_model = nn.Sequential(
            self.model,
            self.new_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the modified ResNet-50 model."""
        return self.combined_model(x)