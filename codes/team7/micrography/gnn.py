import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Optional
import torch.nn.functional as F
import numpy as np

class GraphEmbedder:
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, 
                 learning_rate: float = 0.01, device: str = 'cuda'):
        """
        Initialize the Graph Embedder with autoencoder, clustering, and visualization capabilities.
        
        Args:
            input_dim: Input dimension of node features
            hidden_dim: Hidden layer dimension
            embedding_dim: Dimension of the embedding space
            learning_rate: Learning rate for optimizer
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model(input_dim, hidden_dim, embedding_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Store hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # Initialize trackers
        self.loss_history = []
        self.embeddings = None
        self.labels = None
    
    def _create_model(self, input_dim: int, hidden_dim: int, embedding_dim: int) -> nn.Module:
        """Create and return the GraphAutoencoder model."""
        class GraphAutoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, embedding_dim):
                super().__init__()
                # Encoder
                self.conv1 = GCNConv(input_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, embedding_dim)
                # Decoder
                self.decoder = nn.Linear(embedding_dim, input_dim)
                
                # Add batch normalization and dropout
                self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
                self.batch_norm2 = nn.BatchNorm1d(embedding_dim)
                self.dropout = nn.Dropout(0.2)

            def encode(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = self.batch_norm1(x)
                x = F.relu(x)
                x = self.dropout(x)
                
                x = self.conv2(x, edge_index)
                x = self.batch_norm2(x)
                return x

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x, edge_index):
                z = self.encode(x, edge_index)
                x_reconstructed = self.decode(z)
                return z, x_reconstructed
        
        model = GraphAutoencoder(input_dim, hidden_dim, embedding_dim)
        return model.to(self.device)
    
    def train(self, data, epochs: int = 500, verbose: bool = True) -> np.ndarray:
        """
        Train the model and return embeddings.
        
        Args:
            data: PyG Data object containing x and edge_index
            epochs: Number of training epochs
            verbose: Whether to print progress
        
        Returns:
            numpy array of embeddings
        """
        self.model.train()
        data = data.to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            z, x_reconstructed = self.model(data.x, data.edge_index)
            loss = self.criterion(x_reconstructed, data.x)
            loss.backward()
            self.optimizer.step()
            
            self.loss_history.append(loss.item())
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Store embeddings
        self.embeddings = z.detach().cpu().numpy()
        return self.embeddings
    
    def cluster(self, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        Cluster the embeddings using DBSCAN.
        
        Args:
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
        
        Returns:
            numpy array of cluster labels
        """
        if self.embeddings is None:
            raise ValueError("Must train model before clustering")
        
        self.labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(self.embeddings)
        return self.labels
    
    def visualize(self, show: bool = True, save_path: Optional[str] = None):
        """
        Visualize the embeddings using t-SNE.
        
        Args:
            show: Whether to display the plot
            save_path: Optional path to save the plot
        """
        if self.embeddings is None or self.labels is None:
            raise ValueError("Must train model and cluster before visualizing")
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Compute t-SNE
        tsne = TSNE(n_components=2)
        reduced_embeddings = tsne.fit_transform(self.embeddings)
        
        # Create scatter plot
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                            c=self.labels, cmap='viridis', s=10)
        plt.colorbar(scatter)
        plt.title("t-SNE Visualization of Node Embeddings")
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_loss_history(self, show: bool = True, save_path: Optional[str] = None):
        """Plot training loss history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Training Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_cluster_distribution(self, show: bool = True, save_path: Optional[str] = None):
        """Plot the distribution of cluster sizes."""
        plt.figure(figsize=(10, 6))
        plt.hist(self.labels, bins=np.arange(self.labels.min(), self.labels.max() + 2) - 0.5, edgecolor='black')
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster Size')
        plt.ylabel('Frequency')
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_model(self, path: str):
        """Save model state dict."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim
            }
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cuda'):
        """Load model from saved state dict."""
        checkpoint = torch.load(path)
        
        # Create instance with saved hyperparameters
        instance = cls(
            input_dim=checkpoint['hyperparameters']['input_dim'],
            hidden_dim=checkpoint['hyperparameters']['hidden_dim'],
            embedding_dim=checkpoint['hyperparameters']['embedding_dim'],
            device=device
        )
        
        # Load state dicts
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return instance

    def reset_model(self):
        """Reset the model to its initial state.
        
        This method:
        1. Reinitializes model weights
        2. Resets the optimizer
        3. Clears loss history and embeddings
        """
        # Reinitialize model with same architecture
        self.model = self._create_model(self.input_dim, self.hidden_dim, self.embedding_dim)
        
        # Reset optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Clear stored results
        self.loss_history = []
        self.embeddings = None
        self.labels = None