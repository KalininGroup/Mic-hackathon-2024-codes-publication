import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
from utils.helper import ImageDataset, view_random_image, predict
from torch.utils.data import DataLoader, random_split
from training.training import train, load_model
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# import the images
image_dir='path/to/dataset/X-ray-PBD-main/x_ray_pbd_datasets/train/img'
image_dir2='path/to/dataset/X-ray-PBD-main/x_ray_pbd_datasets/train/crop_mask'

dataset = ImageDataset(image_dir, image_dir2, mean=[0], std=[1]) # no normalization 

# split the dataset into validation and trainign    
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

#Define model parameters 
config = {
    'in_channels': 1,         # Single-channel grayscale input
    'lr': 0.001,              # Lower learning rate for smoother convergence
    'num_epochs': 400,
}

#Training
train(config,train_loader, val_loader)

# Predict 
image_dir_test='/Users/amir.taqieddin/Desktop/Hackathon/X-ray-PBD-main/x_ray_pbd_datasets/test/img'
image_dir_test2='/Users/amir.taqieddin/Desktop/Hackathon/X-ray-PBD-main/x_ray_pbd_datasets/test/crop_mask'
dataset_test = ImageDataset(image_dir_test, image_dir_test2, mean=[0], std=[1]) # no normalization 
# DataLoader for testing
test = DataLoader(dataset_test, batch_size=16, shuffle=False)

# Load the saved model
model = load_model('final_model.pth')

# Make predictions
output = predict(model, test)  # Use only image data for prediction
print("Predictions Shape:", output.shape)  # Shape: [num_samples, 4]

# View random image with bounding box
view_random_image(dataset_test, model)
