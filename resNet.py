import torch                 # Import PyTorch for tensor operations.
import torch.nn as nn        # Import torch.nn to define neural network modules.
from torchvision import models  # Import pre-trained models from torchvision.
from torchvision.models import ResNet50_Weights  # Import weights for ResNet-50.

class FineTunedResNet(nn.Module):
    """
    FineTunedResNet is a modified version of the pre-trained ResNet-50 model for transfer learning.
    
    Key changes:
      - The original fully connected (fc) layer is replaced with a sequential block (self.new_layers) that:
         1. Applies a ReLU activation.
         2. Uses a linear layer to map the 1000 output features to the desired number of classes.
      - This design avoids applying AdaptiveAvgPool2d twice.
      - All layers are frozen except for 'layer4' and 'fc', which are unfreezed for fine-tuning.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Initializes the FineTunedResNet model.
        
        Args:
            num_classes (int): The final number of classes for classification.
        """
        # Call the parent class (nn.Module) constructor.
        super(FineTunedResNet, self).__init__()

        # Load the pre-trained ResNet-50 model with ImageNet weights.
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all layers in the model initially to prevent updating their weights during training.
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze parameters only in "layer4" and "fc", allowing these layers to be fine-tuned.
        for name, layer in self.model.named_children():
            if name in ['layer4', 'fc']:
                for param in layer.parameters():
                    param.requires_grad = True

        # Replace the fully connected (fc) layer with a new sequential block.
        # First, a ReLU activation is applied, then a Linear layer maps the 1000 features to num_classes.
        self.new_layers = nn.Sequential(
            nn.ReLU(),              # Apply ReLU activation.
            nn.Linear(1000, num_classes)  # Map the 1000 features to the required number of classes.
        )

        # Combine the base ResNet-50 model with the new classification head.
        # This sequential module first passes input through the original model,
        # then through the new_layers to produce the final output.
        self.combined_model = nn.Sequential(
            self.model,
            self.new_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the combined model.
        
        Args:
            x (torch.Tensor): Input tensor (e.g. an image or a batch of images).
        
        Returns:
            torch.Tensor: Output tensor containing class scores.
        """
        # Pass the input tensor through the combined model (base ResNet-50 + new classification head).
        return self.combined_model(x)