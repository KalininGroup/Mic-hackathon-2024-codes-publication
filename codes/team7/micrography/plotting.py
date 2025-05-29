import time
import numpy as np
import polars as pl
from pathlib import Path
from PIL import Image
import matplotlib as mpl
import datashader as ds
from datashader.transfer_functions import shade, spread
import colorcet
import matplotlib.pyplot as plt
import networkx as nx

def save_reconstructed_image(vae_regions: np.ndarray, class_data: pl.DataFrame, 
                           sample_dir: Path, name: str, verbose: bool = False):
    """Save reconstructed image with optional benchmarking.
    
    Args:
        vae_regions: Input array of region labels
        class_data: DataFrame containing class predictions
        sample_dir: Directory to save the image
        name: Output filename
        verbose: Whether to print timing information
    """
    timings = {'start': time.perf_counter()}
    
    try:
        import cupy as cp
        # GPU implementation
        timings['transfer_start'] = time.perf_counter()
        vae_regions_dev = cp.asarray(vae_regions)
        
        # Create lookup table
        max_label = int(max(class_data['class'].max(), vae_regions_dev.max().get()))
        lookup = cp.zeros(max_label + 1, dtype=cp.int32)
        for label, pred in zip(class_data['class'], class_data['y_pred']):
            lookup[int(label)] = int(pred)
        timings['transfer_end'] = time.perf_counter()
        
        # Reconstruction
        timings['reconstruct_start'] = time.perf_counter()
        reconstructed_image = lookup[vae_regions_dev]
        reconstructed_image = cp.asnumpy(reconstructed_image)
        timings['reconstruct_end'] = time.perf_counter()
        
    except ImportError:
        # CPU implementation
        timings['reconstruct_start'] = time.perf_counter()
        max_label = max(class_data['class'].max(), vae_regions.max())
        lookup = np.zeros(max_label + 1, dtype=np.int32)
        for label, pred in zip(class_data['class'], class_data['y_pred']):
            lookup[int(label)] = int(pred)
        reconstructed_image = lookup[vae_regions]
        timings['reconstruct_end'] = time.perf_counter()
    
    # Color mapping (always on CPU as it's not the bottleneck)
    timings['color_start'] = time.perf_counter()
    unique_labels = np.unique(reconstructed_image)
    n_colors = len(unique_labels)
    
    # Pre-compute colormap for unique labels
    cm = mpl.cm.nipy_spectral(np.linspace(0, 1, n_colors))
    color_lookup = np.zeros((unique_labels.max() + 1, 4), dtype=np.float32)
    for i, label in enumerate(unique_labels):
        color_lookup[label] = cm[i]
    
    # Vectorized color mapping
    colored_new_labels = color_lookup[reconstructed_image]
    colored_new_labels = (colored_new_labels * 255).astype(np.uint8)
    timings['color_end'] = time.perf_counter()
    
    # Save image
    timings['save_start'] = time.perf_counter()
    img = Image.fromarray(colored_new_labels)
    img.save(sample_dir/f"{name}.png")
    timings['end'] = time.perf_counter()
    
    if verbose:
        total_time = timings['end'] - timings['start']
        print(f"Processing completed on {'GPU' if 'cp' in locals() else 'CPU'}")
        print(f"Total time: {total_time:.4f} seconds")
        
        if 'cp' in locals():
            transfer_time = timings['transfer_end'] - timings['transfer_start']
            print(f"Transfer time: {transfer_time:.4f} seconds")
            
        reconstruct_time = timings['reconstruct_end'] - timings['reconstruct_start']
        color_time = timings['color_end'] - timings['color_start']
        save_time = timings['end'] - timings['save_start']
        print(f"Reconstruction time: {reconstruct_time:.4f} seconds")
        print(f"Color mapping time: {color_time:.4f} seconds")
        print(f"Save time: {save_time:.4f} seconds")

def datashader_plot(df: pl.DataFrame, x: str, y: str, value: str) -> Image.Image:
    cvs = ds.Canvas(plot_width=df[x].max(), plot_height=df[y].max())
    agg = cvs.points(df.to_pandas(), x, y, agg=ds.mean(value))
    img = shade(agg, cmap=colorcet.fire, how='linear')
    img = spread(img, px=5)
    return img.to_pil()

def plot_graph_evolution(feat_target: pl.DataFrame, embeddings: np.ndarray, 
                        labels: np.ndarray, G: nx.Graph, output_dir: Path):
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    # Original graph visualization
    pos = {i: G.nodes[i]['centroid'] for i in G.nodes}
    nx.draw(G, pos, node_color=feat_target['y_pred'], node_size=1, ax=axes[0, 0])
    axes[0, 0].set_title('Original Graph Structure')

    axes[0, 1].scatter(feat_target['median_x'], feat_target['median_y'], 
                        c=labels, cmap='viridis', s=1)
    axes[0, 1].set_title('Node Embeddings (t-SNE)')
    axes[0, 1].axis('off')

    # Embeddings
    axes[1, 0].scatter(embeddings[:, 0], embeddings[:, 1], 
                        c=labels, cmap='viridis', s=1)
    axes[1, 0].set_title('Node Embeddings (t-SNE)')
    axes[1, 0].axis('off')

    # Plot the embeddings with the original graph
    axes[1, 1].scatter(embeddings[:, 0], embeddings[:, 1], 
                      c=feat_target['y_pred'], s=1)
    axes[1, 1].set_title('Node Embeddings with Labeled Molecules')
    axes[1, 1].axis('off')
    fig.savefig(output_dir/"graph_evolution.png")
    plt.tight_layout()
    plt.show()
    