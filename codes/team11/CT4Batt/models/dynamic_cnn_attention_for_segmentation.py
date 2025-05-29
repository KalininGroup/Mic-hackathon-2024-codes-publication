import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention2D(nn.Module):
    def __init__(self, in_channels, ratio=0.25, num_kernels=4, temperature=5, kernel_size=3, activation_fn=nn.ReLU,
                 init_weight=True):
        super(Attention2D, self).__init__()
        assert temperature >= 1, "Temperature must be >= 1."

        # TODO Need to tune this value empirically
        self.temperature = temperature
        hidden_channels = max(int(in_channels * ratio), 1)
        self.activation_fn = activation_fn()

        # Attention layers:
        # We use an AdaptiveAvgPool2d to produce a fixed-size global context (8x8),
        # regardless of the input image size (H,W). This allows the attention mechanism
        # to handle variable input dimensions.
        # TODO Assess what adapative pool size is needed to get enough detail to find the electrodes
        self.global_pool = nn.AdaptiveAvgPool2d((16, 16))

        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden_channels, num_kernels, kernel_size=kernel_size,
                             padding=kernel_size // 2, bias=True)

        # TODO It may be useful to apply BatchNorm or InstanceNorm here
        self.norm = nn.LayerNorm(num_kernels)  # Normalize attention vector for stability

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights for convolution layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x: [B, C, H, W], with variable H and W
        batch_size = x.size(0)

        # Global context extraction
        # The global_pool produces a fixed-size (8x8) spatial map regardless of input size.
        context = self.global_pool(x)  # [B, C, 8, 8]

        # Compute attention features
        attention = self.activation_fn(self.fc1(context))  # [B, hidden, 8, 8]
        attention = self.fc2(attention).mean(dim=(2, 3))  # Average over spatial dims -> [B, num_kernels]

        # Layer normalization and temperature-based softmax
        attention = self.norm(attention)
        attention_weights = F.softmax(attention / (self.temperature + 1e-5), dim=1)  # [B, num_kernels]

        return attention_weights


class DynamicCNN(nn.Module):
    """
    DynamicCNN:
    - Input: [B, 1, H, W] variable H and W
    - The network uses attention (via Attention2D) to compute weights for `num_kernels` filters.
    - These filters are applied dynamically to produce `num_kernels` output masks.
    - Output: [B, num_kernels, H, W], same spatial dimensions as input.
      Each of the num_kernels outputs corresponds to a potential "tab" mask.

    Variable sizing:
    - No fixed input size assumptions. Convolutions and pooling handle arbitrary H and W.
    - Attention uses adaptive pooling, so the network can handle arbitrary spatial dims.
    - Output dimensions match input height and width exactly.
    """

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, ratio=0.25, num_kernels=50, temperature=5,
                 activation_fn=nn.ReLU, init_weight=True):
        super(DynamicCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.num_kernels = num_kernels

        # Attention module: Produces weights (importance) for each of the num_kernels filters.
        self.attention = Attention2D(in_channels, ratio, num_kernels, temperature, kernel_size, activation_fn)

        # Learnable filters for each potential tab.
        # We treat each kernel as a separate tab detector.
        self.weight = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels, kernel_size, kernel_size),
            requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(num_kernels, out_channels))

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        # Initialize the kernel weights and biases
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        # x: [B, in_channels, H, W] - variable H and W
        batch_size, _, height, width = x.size()

        # Compute attention weights for each kernel
        attention_weights = self.attention(x)  # [B, num_kernels]

        # Expand input to apply all kernels separately
        x_expanded = x.repeat(1, self.num_kernels, 1, 1)  # [B, C*num_kernels, H, W]

        # Reshape kernels to match group convolution format
        weight_all = self.weight.view(self.num_kernels * self.out_channels,
                                      self.in_channels,
                                      self.kernel_size,
                                      self.kernel_size)

        # Apply group convolution with num_kernels groups
        out_all = F.conv2d(x_expanded, weight_all, bias=None,
                           padding=self.padding, groups=self.num_kernels)
        # out_all: [B, num_kernels*out_channels, H, W] = [B, num_kernels, H, W] since out_channels=1

        # Add bias per kernel
        bias = self.bias.view(self.num_kernels * self.out_channels)
        bias = bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, num_kernels, 1, 1]
        out_all = out_all + bias

        # Apply attention weights to each kernel's output
        # attention_weights: [B, num_kernels]
        # Reshape to broadcast over spatial dims: [B, num_kernels, 1, 1]
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
        output = out_all * attention_weights

        # Sigmoid to get probability masks per kernel
        output = torch.sigmoid(output)  # [B, num_kernels, H, W]

        return output