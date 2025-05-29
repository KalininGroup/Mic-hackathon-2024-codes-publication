import numpy as np
from skimage.filters import threshold_local
from skimage import img_as_ubyte
import seaborn as sns
import matplotlib as mpl
from sklearn.cluster import DBSCAN
from plotting import save_reconstructed_image
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Dict, Union
import numba
import polars as pl
from scipy.interpolate import Rbf
from networkx import Graph
import time
import platform

# Determine hardware platform and import appropriate acceleration library
if platform.system() == "Darwin" and platform.processor() == "arm":
    try:
        import mlx.core as xp
        ACCEL_DEVICE = "mlx"
    except ImportError:
        import numpy as xp
        ACCEL_DEVICE = "cpu"
else:
    try:
        import cupy as xp
        ACCEL_DEVICE = "cuda"
    except ImportError:
        import numpy as xp
        ACCEL_DEVICE = "cpu"

def plot_and_save(data: np.ndarray, target: Path, name: str, ax: plt.Axes, color: bool = False) -> None:
    """Plots the data and saves the full resolution image to the target directory.

    Args:
        data (np.ndarray): Image array
        target (Path): Where to save the image
        name (str): Plot title
        ax (plt.Axes): The axis to plot on
    """
    print(f"Plotting and saving {name}")
    # Plotting
    ax.imshow(data, cmap='nipy_spectral')
    ax.set_axis_off()
    ax.set_title(name)

    img_data = data
    if color:
        cm = mpl.cm.nipy_spectral(np.linspace(0, 1, len(np.unique(data))))
        colored_labels = np.array([cm[x] for x in data])
        img_data = (colored_labels * 255).astype(np.uint8)  # Convert to uint8

    img = Image.fromarray(img_as_ubyte(img_data))
    img.save(target)


def binarize(data: np.ndarray, block_size: int = 31) -> np.ndarray:
    """Binarize the input data using local thresholding.

    Args:
        data (np.ndarray): Input data.
        block_size (int, optional): Block size for local thresholding. Defaults to 31.

    Returns:
        np.ndarray: Binarized data.
    """
    print("Binarizing data")
    thresh = threshold_local(data, block_size=block_size, method='gaussian', offset=0)
    binary = data > thresh
    return binary


def find_atoms(data: np.ndarray) -> np.ndarray:
    """Find atoms in the binary image.

    Args:
        data (np.ndarray): Binary image.

    Returns:
        np.ndarray: Labeled image.
    """

    print("Finding atoms")
    # Use DBSCAN to label each atom
    coords = np.column_stack(np.nonzero(data))  # Extract coordinates of non-zero pixels
    dbscan = DBSCAN(eps=3, min_samples=5)  # Adjust `eps` and `min_samples` as needed
    label_db = dbscan.fit_predict(coords)  # Cluster the coordinates
    
    # Map DBSCAN labels back to the 2D image
    labels = np.zeros_like(data, dtype=int)  # Initialize the 2D label image
    for coord, label in zip(coords, label_db):
        labels[tuple(coord)] = label + 1  # Avoid 0 for background
    return labels


@numba.njit(fastmath=True, parallel=True)
def find_sizes(labels: np.ndarray, cutoff: int= 200) -> Tuple[np.ndarray, dict, int]:
    """Find the size of each region in the labeled image.

    Args:
        data (np.ndarray): Binary image.
        labels (np.ndarray): Labeled image.

    Returns:
        Tuple[np.ndarray, dict, int]: Data about region sizes.
    """
    sizes = np.bincount(labels.ravel())
    class_sizes = dict(zip(np.unique(labels), sizes))
    class_sizes = {k: v for k, v in class_sizes.items() if v < cutoff}
    num_greater = len(sizes[sizes >= cutoff])
    return sizes, class_sizes, num_greater


@numba.njit(fastmath=True, parallel=True)
def prune_regions(class_sizes: dict, labels: np.ndarray) -> np.ndarray:
    """Prune regions based on class sizes and remap labels to be contiguous.

    Args:
        class_sizes (dict): Dictionary of class sizes.
        labels (np.ndarray): Labeled image.

    Returns:
        np.ndarray: Pruned and remapped labeled image.
    """
    print("Pruning regions")
    pruned_labels = labels.copy()
    keys = np.array(list(class_sizes.keys()))

    for i in numba.prange(pruned_labels.shape[0]):
        for j in range(pruned_labels.shape[1]):
            if pruned_labels[i, j] not in keys:
                pruned_labels[i, j] = 0

    # Remap labels to be contiguous
    unique_labels = np.unique(pruned_labels)
    new_labels = np.arange(len(unique_labels))
    label_map = {old: new for old, new in zip(unique_labels, new_labels)}

    for i in numba.prange(pruned_labels.shape[0]):
        for j in range(pruned_labels.shape[1]):
            pruned_labels[i, j] = label_map[pruned_labels[i, j]]

    return pruned_labels


def class_stats(data: np.ndarray, verbose: bool = False) -> pl.DataFrame:
    """Calculate class statistics using hardware-appropriate acceleration.
    
    Computes statistics for each class in the input data, including:
    - Class label
    - Size (number of pixels)
    - Median x and y positions
    - Standard deviation of x and y positions
    
    Args:
        data: Input labeled image array (2D)
        verbose: Whether to print timing and device information
    
    Returns:
        DataFrame with columns: class, size, median_x, median_y, std_x, std_y
    """
    @numba.njit(parallel=True, fastmath=True)
    def _cpu_class_stats(data: np.ndarray) -> np.ndarray:
        """Numba-accelerated CPU implementation of class statistics calculation."""
        classes = np.unique(data)
        classes = classes[classes != 0]  # Remove background
        n_classes = len(classes)
        all_outputs = np.zeros((n_classes, 6), dtype=np.float32)
        
        for i in numba.prange(n_classes):
            c = classes[i]
            positions = np.where(data == c)
            x_pos = positions[0]
            y_pos = positions[1]
            size = len(x_pos)
            
            median_x = int(np.median(x_pos))
            median_y = int(np.median(y_pos))
            std_x = float(np.std(x_pos))
            std_y = float(np.std(y_pos))
            
            all_outputs[i] = [c, size, median_x, median_y, std_x, std_y]
        
        return all_outputs

    timings: Dict[str, float] = {'start': time.perf_counter()}
    
    try:
        import cupy as cp
        # GPU implementation
        timings['transfer_start'] = time.perf_counter()
        data_dev = cp.asarray(data)
        timings['transfer_end'] = time.perf_counter()
        
        classes = cp.unique(data_dev)
        classes = classes[classes != 0]
        n_classes = len(classes)
        all_outputs = cp.zeros((n_classes, 6), dtype=cp.float32)
        
        timings['compute_start'] = time.perf_counter()
        
        for i, c in enumerate(classes):
            positions = cp.where(data_dev == c)
            x_pos = positions[0]
            y_pos = positions[1]
            size = len(x_pos)
            
            median_x = int(cp.median(x_pos))
            median_y = int(cp.median(y_pos))
            std_x = float(cp.std(x_pos))
            std_y = float(cp.std(y_pos))
            
            all_outputs[i] = [float(c), float(size), median_x, median_y, std_x, std_y]
            
        timings['compute_end'] = time.perf_counter()
        result = cp.asnumpy(all_outputs)
        
    except ImportError:
        # CPU implementation with Numba
        timings['compute_start'] = time.perf_counter()
        result = _cpu_class_stats(data)
        timings['compute_end'] = time.perf_counter()
    
    timings['end'] = time.perf_counter()
    
    # Create DataFrame
    cols_dtypes = {
        'class': pl.Int32, 
        'size': pl.Int32, 
        'median_x': pl.Int32,
        'median_y': pl.Int32,
        'std_x': pl.Float32, 
        'std_y': pl.Float32
    }
    output = pl.DataFrame(result, schema=cols_dtypes)
    
    if verbose:
        total_time = timings['end'] - timings['start']
        print(f"Processing completed on {'GPU' if 'cp' in locals() else 'CPU'}")
        print(f"Total time: {total_time:.4f} seconds")
        
        if 'cp' in locals():
            transfer_time = timings['transfer_end'] - timings['transfer_start']
            compute_time = timings['compute_end'] - timings['compute_start']
            print(f"Transfer time: {transfer_time:.4f} seconds")
            print(f"Compute time: {compute_time:.4f} seconds")
        else:
            compute_time = timings['compute_end'] - timings['compute_start']
            print(f"Compute time: {compute_time:.4f} seconds")
    
    return output


def extract_windows(x: np.ndarray, y: np.ndarray, window_size: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract fixed-size windows from a single image and its labels.
    
    Args:
        x: Input image array
        y: Label image array
        window_size: Size of the window (height=width)
        stride: Stride for window extraction
    
    Returns:
        Tuple of (input windows, label windows) as numpy arrays
    """
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
    num_windows_h = (height - window_size) // stride + 1
    num_windows_w = (width - window_size) // stride + 1
    
    # Check if image is too small
    if num_windows_h <= 0 or num_windows_w <= 0:
        raise ValueError(f"Image shape {x.shape} too small for window_size {window_size}")
    
    # Slide over the image and label to extract windows
    for i in range(num_windows_h):
        for j in range(num_windows_w):
            start_i = i * stride
            start_j = j * stride
            
            # Extract windows
            window_x = x[start_i:start_i+window_size, start_j:start_j+window_size]
            window_y = y[start_i:start_i+window_size, start_j:start_j+window_size]
            
            # Verify window shapes
            if (window_x.shape != (window_size, window_size) or 
                window_y.shape != (window_size, window_size)):
                continue
            
            # Add channel dimension if needed
            window_x = window_x.reshape(window_size, window_size, 1)
            window_y = window_y.reshape(window_size, window_size, 1)
            
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


@numba.njit(fastmath=True, parallel=True)
def reassign_classes(org_img: np.ndarray, org_classes: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Reassign newly learned classes to the original image.

    Args:
        org_img (np.ndarray): A 2D numpy array representing the original image.
        org_classes (np.ndarray): A numpy array representing the original classes.
        y_pred (np.ndarray): A numpy array representing the new classes.

    Returns:
        np.ndarray: A 2D numpy array of the same shape as `org_img` with the new classes.
    """
    class_map = dict(zip(org_classes,y_pred))
    class_map[0] = 0
    output = np.zeros_like(org_img)

    n_row, n_col = org_img.shape
    for row_i in numba.prange(n_row):
        for col_i in numba.prange(n_col):
            output[row_i][col_i] = class_map[org_img[row_i][col_i]]
    return output



def identify_molecules(sample, vae_img, vae_class_features, vae_regions, target):
    output_fmt = 'data/stats_df_vae_{sample}.parquet'
    sample_dir = target/f"sample_{sample}"
    vae_regions[sample] = find_atoms(binarize(vae_img[sample]))

    cm = mpl.cm.nipy_spectral(np.linspace(0, 1, np.unique(vae_regions[sample]).size))
    colored_new_labels = np.array([cm[label] for label in vae_regions[sample]])
    colored_new_labels = (colored_new_labels * 255).astype(np.uint8)  # Convert to uint8

    img = Image.fromarray(colored_new_labels)
    img.save(sample_dir/"vae_regions.png")

    # Plot the vae segmented image and the regions
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(vae_img[sample], cmap='nipy_spectral')
    axes[0].set_title('VAE Segmented Image')
    axes[0].axis('off')
    axes[1].imshow(vae_regions[sample], cmap='nipy_spectral')
    axes[1].set_title('VAE Regions')
    axes[1].axis('off')
    plt.show()

    output = Path(output_fmt.format(sample=sample))
    cached = output.exists()
    print(f"Using cached data for sample {sample}: {cached}")
    if not cached:
        class_data = class_stats(vae_regions[sample], verbose=True)

        # Remove the background class
        bg_class, bg_count = np.unique(vae_regions[sample].flatten(), return_counts=True)
        bg_class = bg_class[np.argmax(bg_count)]

        org_classes = transfer_classes(
            class_data.filter(pl.col('class') != bg_class),
            vae_img[sample], 
            verbose=True
        )
        class_data = class_data.join(org_classes, on='class')
        class_data.write_parquet(output)
    else:
        class_data = pl.read_parquet(output)
    save_reconstructed_image(
        vae_regions[sample], 
        class_data, 
        sample_dir, 
        'vae_reconstructed', 
        verbose=True
    )
    vae_class_features[sample] = class_data

    fig, axes = plt.subplots(2, len(class_data.columns)//2, figsize=(18, 9))
    for i, (col, ax) in enumerate(zip(class_data.columns, axes.flatten())):
        sns.histplot(class_data[col], ax=ax, kde=True)
        ax.set_title(col)



### GRAPH FUNCTIONS

@numba.njit(fastmath=True, parallel=True)
def get_nearest_neighbors(centroids: np.ndarray, k: int = 4):
    dists = np.zeros((centroids.shape[0], k))
    indices = np.zeros((centroids.shape[0], k))
    for i in numba.prange(centroids.shape[0]):
        dist = np.sqrt(np.sum((centroids - centroids[i]) ** 2, axis=1))
        sorted_indices = np.argsort(dist)[1:k+1]  # Skip the first one because it's the point itself
        dists[i] = dist[sorted_indices]
        indices[i] = sorted_indices
    return dists, indices.astype(np.uint32)

def find_neighbors(class_features: pl.DataFrame, k: int = 4):
    centroids = np.array(
        [
            class_features['median_x'].to_numpy(), 
            class_features['median_y'].to_numpy()
        ]
    ).T
    dists, indices = get_nearest_neighbors(centroids, k)

    # Let's also add the 4 nearest neighbors to the class features
    class_features = class_features.with_columns([pl.Series(f'nn_{i+1}_dist', dists[:, i]) for i in range(k)])
    return (dists, indices), class_features

def construct_graph(dists, indices, class_feat) -> Graph:
    G = Graph()
    centroids = class_feat.select(['median_x', 'median_y']).to_numpy()
    for i, (dist, idx) in enumerate(zip(dists, indices)):
        for d, j in zip(dist, idx):
            G.add_edge(i, j, weight=d)
        
        G.nodes[i]['centroid'] = centroids[i]
        G.nodes[i]['size'] = class_feat['size'][i]
        G.nodes[i]['std_x'] = class_feat['std_x'][i]
        G.nodes[i]['std_y'] = class_feat['std_y'][i]
        G.nodes[i]['class'] = class_feat['y_pred'][i]
    return G

def interpolate_and_plot(df, x_col, y_col, value_col, ax, grid_size=4000, resolution=500, rbf_function='multiquadric'):
    """
    Interpolates data points in a 2D space using Radial Basis Function (RBF) and plots the interpolated image.

    Parameters:
        df (pd.DataFrame): DataFrame containing the input data.
        x_col (str): Name of the column for x-coordinates.
        y_col (str): Name of the column for y-coordinates.
        value_col (str): Name of the column for the continuous values to interpolate.
        grid_size (int): Maximum size of the space (assumes square grid). Default is 4000.
        resolution (int): Number of grid points per axis (lower for faster computation). Default is 500.
        rbf_function (str): RBF function to use. Options: 'multiquadric', 'linear', 'cubic', etc.

    Returns:
        np.ndarray: Interpolated grid of values.
    """
    # Extract columns from DataFrame
    x = df[x_col].values
    y = df[y_col].values
    values = df[value_col].values

    # Create a grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, grid_size, resolution), 
        np.linspace(0, grid_size, resolution)
    )

    # RBF Interpolation
    rbf = Rbf(x, y, values, function=rbf_function)
    interpolated_values = rbf(grid_x, grid_y)

    # Plot the interpolated image
    ax.imshow(
        interpolated_values, 
        extent=(0, grid_size, 0, grid_size), 
        origin='lower', 
        cmap='viridis', 
        alpha=0.85
    )
    ax.scatter(x, y, c=values, cmap='viridis', s=20, edgecolor='k', label='Input Points')
    ax.title(f"RBF Interpolation: {value_col}")
    ax.set_axis_off()


def transfer_classes(class_data: pl.DataFrame, y_pred: np.ndarray, verbose: bool = False) -> pl.DataFrame:
    """Transfer predicted classes to molecules with optional benchmarking.

    Args:
        class_data: DataFrame containing molecule data
        y_pred: Predicted classes array
        verbose: Whether to print timing information

    Returns:
        DataFrame with transferred classes
    """
    @numba.njit(parallel=True, fastmath=True)
    def _test_assign_chunked(molecules_x, molecules_y, y_pred):
        n = len(molecules_x)
        result = np.empty(n, dtype=y_pred.dtype)
        chunk_size = 1000
        
        for chunk in numba.prange((n + chunk_size - 1) // chunk_size):
            start = chunk * chunk_size
            end = min(start + chunk_size, n)
            for i in range(start, end):
                result[i] = y_pred[molecules_x[i], molecules_y[i]]
        
        return result

    timings: Dict[str, float] = {'start': time.perf_counter()}
    
    # Make sure we have writable copies of our input arrays
    timings['copy_start'] = time.perf_counter()
    molecules = class_data['class'].to_numpy()
    molecules_x = class_data['median_x'].to_numpy().astype(np.int32)
    molecules_y = class_data['median_y'].to_numpy().astype(np.int32)
    y_pred = y_pred.copy()
    timings['copy_end'] = time.perf_counter()
    
    try:
        import cupy as cp
        # GPU implementation
        timings['transfer_start'] = time.perf_counter()
        molecules_x_dev = cp.asarray(molecules_x)
        molecules_y_dev = cp.asarray(molecules_y)
        y_pred_dev = cp.asarray(y_pred)
        timings['transfer_end'] = time.perf_counter()
        
        timings['compute_start'] = time.perf_counter()
        idx = cp.arange(len(molecules_x))
        output_dev = y_pred_dev[molecules_x_dev[idx], molecules_y_dev[idx]]
        output = cp.asnumpy(output_dev)
        timings['compute_end'] = time.perf_counter()
        
    except ImportError:
        # CPU implementation with Numba
        timings['compute_start'] = time.perf_counter()
        output = _test_assign_chunked(molecules_x, molecules_y, y_pred)
        timings['compute_end'] = time.perf_counter()
    
    # DataFrame creation timing
    timings['df_start'] = time.perf_counter()
    result = pl.DataFrame({
        'class': molecules,
        'y_pred': output
    })
    timings['end'] = time.perf_counter()

    if verbose:
        total_time = timings['end'] - timings['start']
        print(f"Processing completed on {'GPU' if 'cp' in locals() else 'CPU'}")
        print(f"Total time: {total_time:.4f} seconds")
        
        if 'cp' in locals():
            transfer_time = timings['transfer_end'] - timings['transfer_start']
            compute_time = timings['compute_end'] - timings['compute_start']
            print(f"Transfer time: {transfer_time:.4f} seconds")
            print(f"Compute time: {compute_time:.4f} seconds")
        else:
            compute_time = timings['compute_end'] - timings['compute_start']
            print(f"Compute time: {compute_time:.4f} seconds")
        
        copy_time = timings['copy_end'] - timings['copy_start']
        df_time = timings['end'] - timings['df_start']
        print(f"Copy time: {copy_time:.4f} seconds")
        print(f"DataFrame creation time: {df_time:.4f} seconds")
    
    return result
