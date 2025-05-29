import os
import sys
# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch.nn as nn
from models.kan_linear import KANLinear
from models.cnn_attention import attention, SlidingWindowAttention, ResidualBlock

class CNN2DKan(nn.Module):
    """
    Dynamic 2D Convolution with attention-based kernel selection.
    """
    def __init__(self, in_channels=1, base=64):
        super(CNN2DKan, self).__init__()
        # Initial Convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(base),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Coordinate Attention
        self.coord_attention = attention(base)

        # Sliding Window Attention
        self.window_attention = SlidingWindowAttention(base, window_size=7)

        # Residual Feature Extractor
        self.res_blocks = nn.Sequential(
            ResidualBlock(base, base * 2, stride=2),
            ResidualBlock(base * 2, base * 4, stride=2),
            ResidualBlock(base * 4, base * 4)
        )

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # KANLinear-based Bounding Box Prediction
        self.kan_predictor = nn.Sequential(
            nn.Flatten(),
            KANLinear(base * 4 * 4 * 4, 512),  # Replace with KANLinear
            nn.SiLU(),
            nn.Dropout(0.3),
            KANLinear(512, 4)  # Final layer for bounding box regression
        )

    def forward(self, x):
        x = self.initial_conv(x)  # Initial Convolution
        x = self.coord_attention(x)  # Apply Coordinate Attention
        x = self.window_attention(x)  # Sliding Window Attention
        x = self.res_blocks(x)  # Residual Blocks
        x = self.global_pool(x)  # Global Pooling
        x = self.kan_predictor(x)  # Bounding box regression
        return x