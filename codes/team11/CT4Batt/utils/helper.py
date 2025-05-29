import os
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch 
import random
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def read_and_resize(folder_path):
    """Reads images from a folder and applies a transformation."""

    transformed_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".JPG", ".BMP",".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)

            # Apply transformation here
            img = img.resize((256, 256))
            img = img.convert("L") 
            # Convert to numpy array
            img_array = np.array(img)
            
            transformed_images.append(img_array)

    return transformed_images

# Read and resize the images
class ImageDataset(Dataset):
    def __init__(self, org_dir, crop_dir, mean=None, std=None):
        self.mean = mean
        self.std = std

        # Exclude original images with 'jpg' extension for training on less data
        self.org_files = {os.path.splitext(f)[0]: os.path.join(org_dir, f)
                          for f in os.listdir(org_dir)
                          if not f.lower().endswith(".jpg") and f.lower().endswith((".jpeg", ".bmp", ".png"))}

        # Keep only matching cropped files
        self.crop_files = {os.path.splitext(f)[0]: os.path.join(crop_dir, f)
                           for f in os.listdir(crop_dir)
                           if f.lower().endswith((".jpeg", ".bmp", ".png"))}

        self.common_keys = sorted(set(self.org_files.keys()) & set(self.crop_files.keys()))

        if len(self.common_keys) == 0:
            raise ValueError("No matching filenames found between original and cropped images.")

        self.target_size = (256, 256)  # Target resize dimensions

    def __len__(self):
        return len(self.common_keys)

    def __getitem__(self, idx):
        key = self.common_keys[idx]

        org_path = self.org_files[key]
        crop_path = self.crop_files[key]

        # Load original and cropped images
        image_org = Image.open(org_path).convert("L")
        image_crop = Image.open(crop_path).convert("L")

        # Get original dimensions of the cropped image
        W_orig, H_orig = image_crop.size  # Original size before resizing

        # Extract bounding box from the original cropped image
        vertices = self.extract_bounding_box(np.array(image_crop))

        # Rescale bounding box to target size
        if vertices is not None:
            x_min, y_min, x_max, y_max = vertices
            x_min = x_min * self.target_size[0] / W_orig
            x_max = x_max * self.target_size[0] / W_orig
            y_min = y_min * self.target_size[1] / H_orig
            y_max = y_max * self.target_size[1] / H_orig
            vertices = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)  # Convert to tensor
        else:
            vertices = torch.tensor([0, 0, 0, 0], dtype=torch.float32)  # Default box if no vertices are found

        # Resize images to target size and normalize pixel values to [0, 1]
        image_org = image_org.resize(self.target_size)
        image_crop = image_crop.resize(self.target_size)
        
        image_org_np = np.array(image_org) / 255.0  # Normalize to [0, 1]
        image_crop_np = np.array(image_crop) / 255.0  # Normalize to [0, 1]

        # Convert images to tensors
        image_org_tensor = torch.tensor(image_org_np, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        image_crop_tensor = torch.tensor(image_crop_np, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image_org_tensor, vertices

    @staticmethod
    def extract_bounding_box(image):
        """
        Extract bounding box coordinates (vertices) from the non-zero pixel region of a grayscale image.
        Args:
            image: Grayscale image (2D numpy array).
        Returns:
            List of (x_min, y_min, x_max, y_max) coordinates.
        """
        # Threshold the image (to find the non-zero region)
        _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None  # No non-zero region found

        # Get the bounding boxes around all contours
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

        # Sort bounding boxes by area (largest to smallest)
        bounding_boxes = sorted(bounding_boxes, key=lambda x: x[2] * x[3], reverse=True)

        # Extract the largest bounding box
        x, y, w, h = bounding_boxes[0]
        return [x, y, x + w, y + h]  # Return (x_min, y_min, x_max, y_max)


class normalize_img(Dataset):
    def __init__(self, images_org, images_crop, mean=None, std=None):
        self.images_org = images_org
        self.images_crop = images_crop
        self.mean = mean
        self.std = std

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean] if mean else [0.0], 
                                 std=[std] if std else [1.0])
        ])
        
    def __len__(self):
        return len(self.images_org)
    
    def __getitem__(self, idx):
        image_org = self.images_org[idx]
        image_crop = self.images_crop[idx]

        if isinstance(image_org, str):
            image_org = Image.open(image_org).convert("L")
        if isinstance(image_crop, str):
            image_crop = Image.open(image_crop).convert("L")
            image_crop = (image_crop > 0.5).float()

        image_org = self.transform(image_org)
        image_crop = self.transform(image_crop)

        
        return image_org, image_crop
    
def view_random_image(dataset, model, device=None):
    """
    Display a random image with predicted bounding box.

    Args:
        dataset: Dataset containing images.
        model: Trained model for prediction.
        device: Device to perform inference.
    """
    device=torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.eval()
    idx = random.randint(0, len(dataset) - 1)  # Randomly select an index
    img_org, vertices = dataset[idx]  # Get the image and ground truth bounding box

    # Preprocess image for model
    image_tensor = img_org.unsqueeze(0).to(device)  # Add batch dimension
    prediction = model(image_tensor).cpu().detach().numpy()[0]  # Get prediction
    gt_vertices = vertices.cpu().detach().numpy()

    # Denormalize ground truth and prediction (if applicable)
    H, W = img_org.shape[1], img_org.shape[2]


    # Clamp and validate coordinates
    x_min_pred, y_min_pred = max(0, int(prediction[0])), max(0, int(prediction[1]))
    x_max_pred, y_max_pred = min(W, int(prediction[2])), min(H, int(prediction[3]))

    x_min_gt, y_min_gt = max(0, int(vertices[0])), max(0, int(vertices[1]))
    x_max_gt, y_max_gt = min(W, int(vertices[2])), min(H, int(vertices[3]))

    # Convert the original image to numpy array
    image_np = img_org.permute(1, 2, 0).cpu().detach().numpy()
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

    # Create copies for prediction and ground truth
    image_gt = image_np.copy()
    image_combined = image_np.copy()

    # Draw predicted rectangle
    image_gt = image_np.copy()
    cv2.rectangle(image_combined, (x_min_pred, y_min_pred), (x_max_pred, y_max_pred), (0, 255, 0), thickness=2)

    # Draw ground truth rectangle
    cv2.rectangle(image_gt, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (255, 0, 0), thickness=2)
    cropped_region = image_np[y_min_pred:y_max_pred, x_min_pred:x_max_pred]
    if cropped_region.size > 0:  # Ensure the crop is valid
        zoomed_image = cv2.resize(cropped_region, (256, 256), interpolation=cv2.INTER_LINEAR)
    else:
        zoomed_image = np.zeros((256, 256, 3), dtype=np.uint8)  # Blank image if crop is invalid

    # Plot the two images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_gt)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')

    axes[1].imshow(image_combined)
    axes[1].set_title("Prediction")
    axes[1].axis('off')

    axes[2].imshow(zoomed_image)
    axes[2].set_title("Zoomed Region (Predicted Box)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_predictions(outputs, targets, original_images, idx=0):
    """
    Visualize the original image with predicted and ground truth rectangles.

    Args:
        outputs (torch.Tensor): Predicted vertices, shape (N, 4).
        targets (torch.Tensor): Ground truth vertices, shape (N, 4).
        original_images (torch.Tensor): Original images, shape (N, C, H, W).
        idx (int): Index of the image to visualize in the batch.
    """
    if idx >= outputs.size(0):
        raise ValueError(f"Index {idx} is out of range for batch size {outputs.size(0)}")

    # Extract predicted and ground truth vertices
    pred_vertices = outputs[idx].cpu().detach().numpy()
    gt_vertices = targets[idx].cpu().detach().numpy()

    # Image dimensions
    H, W = original_images.shape[2], original_images.shape[3]

    # Clamp and validate coordinates
    x_min_pred, y_min_pred = max(0, int(pred_vertices[0])), max(0, int(pred_vertices[1]))
    x_max_pred, y_max_pred = min(W, int(pred_vertices[2])), min(H, int(pred_vertices[3]))

    x_min_gt, y_min_gt = max(0, int(gt_vertices[0])), max(0, int(gt_vertices[1]))
    x_max_gt, y_max_gt = min(W, int(gt_vertices[2])), min(H, int(gt_vertices[3]))

    # Convert the original image to numpy array
    image_np = original_images[idx].permute(1, 2, 0).cpu().detach().numpy()
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

    # Create copies for prediction and ground truth
    image_gt = image_np.copy()
    image_combined = image_np.copy()

    # Draw predicted rectangle
    image_gt = image_np.copy()
    cv2.rectangle(image_combined, (x_min_pred, y_min_pred), (x_max_pred, y_max_pred), (0, 255, 0), thickness=2)

    # Draw ground truth rectangle
    cv2.rectangle(image_gt, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (255, 0, 0), thickness=2)
    cropped_region = image_np[y_min_pred:y_max_pred, x_min_pred:x_max_pred]
    if cropped_region.size > 0:  # Ensure the crop is valid
        zoomed_image = cv2.resize(cropped_region, (256, 256), interpolation=cv2.INTER_LINEAR)
    else:
        zoomed_image = np.zeros((256, 256, 3), dtype=np.uint8)  # Blank image if crop is invalid

    # Plot the two images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_gt)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')

    axes[1].imshow(image_combined)
    axes[1].set_title("Prediction")
    axes[1].axis('off')

    axes[2].imshow(zoomed_image)
    axes[2].set_title("Zoomed Region (Predicted Box)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def predict(model, dataloader, device=torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))):
    """
    Make predictions using the trained model on test data.

    Args:
        model: The trained model.
        dataloader: DataLoader for the test dataset.
        device: Device to run predictions on (CPU or GPU).

    Returns:
        predictions: List of predicted bounding boxes.
    """
    model.eval()
    predictions = []
    device = torch.device(device)

    model.to(device)
    with torch.no_grad():
        for batch in dataloader:
            img_org = batch[0].to(device)  # Extract only the image part (img_org)
            outputs = model(img_org)  # Forward pass
            predictions.append(outputs.cpu())  # Move outputs to CPU for further use

    return torch.cat(predictions, dim=0)  # Combine all predictions