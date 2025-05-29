import numpy as np
import torch.nn as nn


class NormalizeImage(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        x = sample['x']

        # Compute mean and std for each slice
        xmean = np.mean(x, axis=(0, 1, 2), keepdims=True)
        xstd = np.std(x, axis=(0, 1, 2), keepdims=True)

        x = (x - xmean) / xstd

        sample['x'] = x

        return sample
