import numpy as np
import torch
from cellSAM import segment_cellular_image

if __name__ == "__main__":
    img = np.load("/app/yeaz.npy")  # H, W, C
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask, embedding, bounding_boxes = segment_cellular_image(img, device=str(device))
    print("Warmup (download weights) complete")
