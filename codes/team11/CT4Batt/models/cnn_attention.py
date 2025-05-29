import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation Attention Mechanism
class attention(nn.Module):
    """ Squeeze-and-Excitation Attention Mechanism """
    def __init__(self, in_channels, reduction=16):
        super(attention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden_channels = in_channels // reduction

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.act = nn.SiLU()  # Swish activation
        self.conv_h = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # Attention on Height
        x_h = self.pool_h(x).permute(0, 1, 3, 2)  # Shape: (B, C, 1, W)
        x_w = self.pool_w(x)                      # Shape: (B, C, H, 1)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.conv1(y))

        split_size = y.size(2) // 2  # Determine the size to split in half
        x_h = self.sigmoid(self.conv_h(y[:, :, :split_size, :].permute(0, 1, 3, 2)))  # First half (Height)
        x_w = self.sigmoid(self.conv_w(y[:, :, split_size:, :]))  # Second half (Width)

        return x * x_h.expand_as(x) * x_w.expand_as(x)

# Sliding Window Attention
class SlidingWindowAttention(nn.Module):
    """ Sliding Window Attention: Local attention mechanism over windows """
    def __init__(self, in_channels, window_size=7):
        super(SlidingWindowAttention, self).__init__()
        self.window_size = window_size
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=window_size, stride=1, padding=window_size // 2, groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)
    
# Spatial filter followed by feature mixing
class DepthwiseSeparableConv(nn.Module):
    """ Depthwise Separable Convolution """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))

# Encoder based on self attention
class TransformerEncoderBlock(nn.Module):
    """ Lightweight Transformer Encoder Block """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        identity = x
        x = self.self_attn(x, x, x)[0]
        x = self.norm1(x + identity)

        identity = x
        x = self.ffn(x)
        x = self.norm2(x + identity)
        return x

# Downsampling Residual Block
class ResidualBlock(nn.Module):
    """ Residual Block """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + identity)

# 2D CNN with MLP
class CNN2D(nn.Module):
    def __init__(self, in_channels=1, base=64):
        super(CNN2D, self).__init__()

        # Initial Convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(base),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Coordinate Attention
        self.coord_attention = attention(base)

        self.window_attention = SlidingWindowAttention(base, window_size=7)

        # Residual Feature Extractor
        self.res_blocks = nn.Sequential(
            ResidualBlock(base, base * 2, stride=2),
            ResidualBlock(base * 2, base * 4, stride=2),
            ResidualBlock(base * 4, base * 4)
        )

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # MLP for Bounding Box Prediction
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(base * 4 * 4 * 4),
            nn.Linear(base * 4 * 4 * 4, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)  # Predict bounding box: x_min, y_min, x_max, y_max
        )

    def forward(self, x):
        x = self.initial_conv(x)  # Initial Convolution
        x = self.coord_attention(x)  # Apply Coordinate Attention
        x = self.window_attention(x)  # Sliding Window Attention
        x = self.res_blocks(x)  # Residual Blocks
        x = self.global_pool(x)  # Global Pooling
        x = self.mlp(x)  # Bounding box regression
        return x
