# Define the Autoencoder model
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from pathlib import Path
import numpy as np
from typing import Tuple
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, num_classes, latent_dim=64, patch_size=32, stride=16):
        super(Autoencoder, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.stride = stride
        
        # Initialize model architecture
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size * patch_size, 256),  # Use patch_size instead of hardcoded 32
            nn.GELU(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, patch_size * patch_size * num_classes),  # Use patch_size
            nn.Unflatten(1, (num_classes, patch_size, patch_size))  # Use patch_size
        )
        
        # Initialize training components as None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.device = None
        
        # Training history
        self.learning_rates = []
        self.losses = []
        self.accuracies = []

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def compile(self, device='cuda', learning_rate=0.001):
        """Initialize training components."""
        self.device = device
        self.to(device)
        
        # Initialize optimizer with fused=False for MPS
        self.optimizer = optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4,
            fused=False if str(device) == 'mps' else True
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=device=='cuda')
    
    def reset_model(self):
        """Reset the model to its initial state."""
        # Store current configuration
        device = self.device
        learning_rate = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001
        
        # Reinitialize model weights
        def reset_parameters(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        
        self.apply(reset_parameters)
        
        # Recompile with same settings
        self.compile(device=device, learning_rate=learning_rate)
        
        # Clear history
        self.learning_rates = []
        self.losses = []
        self.accuracies = []
    
    def train_epoch(self, dataloader):
        """Train the model for one epoch."""
        self.train()
        total_loss = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Handle mixed precision training based on device
            if self.device == "cuda":
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Scale loss and compute gradients
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # CPU or MPS - use regular training
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def compute_accuracy(self, dataloader):
        """Compute accuracy on the given dataloader."""
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self(inputs)
                preds = torch.argmax(outputs, dim=1)  # Multiclass prediction
                correct += (preds == targets).sum().item()
                total += targets.numel()
        return correct / total
    
    def extract_windows(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract fixed-size windows from a single image and its labels."""
        # Initialize lists to hold the windows
        windows_x = []
        windows_y = []
        
        # Ensure images are properly shaped
        if x is None or y is None:
            raise ValueError("Input image or labels are None")
            
        # Handle NaN values
        x = np.nan_to_num(x, nan=0)
        y = np.nan_to_num(y, nan=0)
        
        # Get dimensions
        height, width = x.shape
        
        # Calculate the number of windows in height and width
        num_windows_h = (height - self.patch_size) // self.stride + 1
        num_windows_w = (width - self.patch_size) // self.stride + 1
        
        # Check if image is too small
        if num_windows_h <= 0 or num_windows_w <= 0:
            raise ValueError(f"Image shape {x.shape} too small for window_size {self.patch_size}")
        
        # Slide over the image and label to extract windows
        for i in range(num_windows_h):
            for j in range(num_windows_w):
                start_i = i * self.stride
                start_j = j * self.stride
                
                # Extract windows
                window_x = x[start_i:start_i+self.patch_size, start_j:start_j+self.patch_size]
                window_y = y[start_i:start_i+self.patch_size, start_j:start_j+self.patch_size]
                
                # Verify window shapes
                if (window_x.shape != (self.patch_size, self.patch_size) or 
                    window_y.shape != (self.patch_size, self.patch_size)):
                    continue
                
                # Add channel dimension if needed
                window_x = window_x.reshape(self.patch_size, self.patch_size, 1)
                window_y = window_y.reshape(self.patch_size, self.patch_size, 1)
                
                # Append to the list of windows
                windows_x.append(window_x)
                windows_y.append(window_y)
        
        if not windows_x:
            raise ValueError("No valid windows were extracted!")
        
        # Convert lists to numpy arrays with explicit shapes
        output_x = np.stack(windows_x)  # Shape: (N, window_size, window_size, 1)
        output_y = np.stack(windows_y)  # Shape: (N, window_size, window_size, 1)
        
        print(f"Extracted {len(windows_x)} windows with shapes: x={output_x.shape}, y={output_y.shape}")
        
        return output_x, output_y

    def visualize_windows(self, img_windows_x: np.ndarray, img_windows_y: np.ndarray, num_samples: int = 5) -> plt.Figure:
        """
        Visualize random windows from the dataset along with their labels.
        
        Args:
            img_windows_x (np.ndarray): Array of image windows
            img_windows_y (np.ndarray): Array of corresponding label windows
            num_samples (int): Number of random windows to display
            
        Returns:
            plt.Figure: Figure containing the visualization
        """
        # Ensure we don't try to sample more windows than we have
        num_samples = min(num_samples, len(img_windows_x))
        random_indices = np.random.choice(len(img_windows_x), num_samples, replace=False)

        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        for i, idx in enumerate(random_indices):
            # Image window
            axes[0, i].imshow(img_windows_x[idx], cmap='viridis')
            axes[0, i].set_title(f"Image {i+1}")
            axes[0, i].axis('off')
            
            # Label window
            axes[1, i].imshow(img_windows_y[idx])
            axes[1, i].set_title(f"Label {i+1}")
            axes[1, i].axis('off')

        plt.tight_layout()
        return fig

    def visualize_latent_dist(self, dataloader, reduction='pca', num_points=1000) -> plt.Figure:
        """
        Visualize the latent space of the Autoencoder.
        
        Args:
            dataloader (DataLoader): DataLoader containing input data and labels.
            reduction (str): Dimensionality reduction method ('pca' or 'tsne').
            num_points (int): Number of points to sample for visualization.
            
        Returns:
            plt.Figure: A scatter plot of the latent space.
        """
        self.eval()
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                # Flatten targets if needed
                targets = targets.flatten().to(self.device)
                latent = self.encoder(inputs)  # Extract latent representations
                latent_vectors.append(latent.cpu())
                labels.append(targets.cpu())
                
        # Combine all batches into a single tensor
        latent_vectors = torch.cat(latent_vectors, dim=0)
        labels = torch.cat(labels, dim=0)
        
        # Subsample if needed
        if latent_vectors.shape[0] > num_points:
            idx = torch.randperm(latent_vectors.shape[0])[:num_points]
            latent_vectors = latent_vectors[idx]
            labels = labels[idx]

        # Convert to numpy for visualization
        latent_vectors = latent_vectors.numpy()
        labels = labels.numpy()
        
        # Apply dimensionality reduction
        if reduction == 'pca':
            reducer = PCA(n_components=2)
        elif reduction == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        else:
            raise ValueError("Invalid reduction method. Use 'pca' or 'tsne'.")
        
        reduced_latent = reducer.fit_transform(latent_vectors)

        # Plot the latent space
        fig, ax = plt.subplots(1,1, figsize=(12, 8))
        scatter = ax.scatter(
            reduced_latent[:, 0], 
            reduced_latent[:, 1], 
            c=labels,
            cmap='tab10',
            alpha=0.7
        )
        fig.colorbar(scatter, label='Labels')
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        fig.suptitle(f'Latent Space Visualization ({reduction.upper()})')
        return fig

    def visualize_latent_space(self, grid_size=10, latent_range=(-3, 3), image_size=(32, 32)) -> plt.Figure:
        """
        Visualize the decoded images from a grid of points in the latent space.
        
        Args:
            grid_size (int): Number of points along each dimension of the grid.
            latent_range (tuple): Range of values for the latent space grid.
            image_size (tuple): Size of the decoded images (H, W).
            
        Returns:
            plt.Figure: Displays a grid of decoded images.
        """
        self.eval()
        
        # Create a grid of latent points
        latent_points = np.linspace(latent_range[0], latent_range[1], grid_size)
        grid_x, grid_y = np.meshgrid(latent_points, latent_points)
        latent_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        
        # Ensure the latent grid has the correct shape
        if latent_grid.shape[1] != self.latent_dim:
            latent_grid = np.pad(latent_grid, ((0, 0), (0, self.latent_dim - latent_grid.shape[1])), 'constant')
        
        latent_grid = torch.tensor(latent_grid, dtype=torch.float32).to(self.device)
        
        # Decode latent points to generate images
        with torch.no_grad():
            decoded_images = self.decoder(latent_grid)
        
        # Move decoded images to CPU and reshape
        decoded_images = decoded_images.cpu().numpy()
        decoded_images = decoded_images.reshape(-1, *image_size)  # Assuming single-channel
        
        # Create a grid to plot the images
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        for i in range(grid_size):
            for j in range(grid_size):
                img_idx = i * grid_size + j
                axes[i, j].imshow(decoded_images[img_idx], cmap='gray')
                axes[i, j].axis('off')
        
        fig.suptitle("Decoded Images from Latent Space Grid", fontsize=16)
        fig.subplots_adjust(wspace=0, hspace=0)
        return fig

    def segment_image(self, image, y_pred, sample, target: Path):
        """
        Segment an image using the trained autoencoder.
        
        Args:
            image: Input image to segment
            y_pred: Predicted labels from another method (e.g., GMM)
            sample: Sample identifier
            target (Path): Path to save the segmentation results
            
        Returns:
            np.ndarray: Segmented image
        """
        self.eval()
        with torch.no_grad():
            # Process the full image
            full_image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            height, width = full_image.shape[2:]
            segmented_image = torch.zeros_like(full_image)
            
            for row_start in range(0, height - self.patch_size + 1, self.stride):
                for col_start in range(0, width - self.patch_size + 1, self.stride):
                    # Extract patch
                    patch = full_image[:, :, row_start:row_start+self.patch_size, col_start:col_start+self.patch_size]
                    
                    # Flatten the patch to match training dimensions
                    batch_size = patch.size(0)
                    patch_flat = patch.reshape(batch_size, self.patch_size * self.patch_size)
                    
                    # Process patch
                    try:
                        output = self(patch_flat)
                        # If output is logits (num_classes, H, W), get predicted class
                        if output.dim() == 4:  # Shape: (batch_size, num_classes, H, W)
                            output = torch.argmax(output, dim=1, keepdim=True)
                        elif output.dim() == 2:  # Shape: (batch_size, patch_size*patch_size)
                            output = output.reshape(batch_size, 1, self.patch_size, self.patch_size)
                    except RuntimeError as e:
                        print(f"Error processing patch at ({row_start}, {col_start}): {e}")
                        continue
                        
                    # Place processed patch back
                    segmented_image[:, :, row_start:row_start+self.patch_size, col_start:col_start+self.patch_size] = output
            
            # Convert to numpy and visualize
            segmented = segmented_image.cpu().squeeze().numpy().astype(np.uint8)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(segmented, cmap='nipy_spectral')
            axes[1].set_title('VAE Image')
            axes[1].axis('off')
            
            axes[2].imshow(y_pred, cmap='nipy_spectral')
            axes[2].set_title('GMM Labels')
            axes[2].axis('off')

            plt.show()

            # Save the segmented image
            cm = mpl.cm.nipy_spectral(np.linspace(0, 1, len(np.unique(segmented))))
            colored_new_labels = np.array([cm[label] for label in segmented])
            colored_new_labels = (colored_new_labels * 255).astype(np.uint8)

            img = Image.fromarray(colored_new_labels)
            img.save(target/f"vae_classes.png")
            return segmented