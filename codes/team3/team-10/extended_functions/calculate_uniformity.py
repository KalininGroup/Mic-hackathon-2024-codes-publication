import numpy as np
from sklearn.neighbors import NearestNeighbors


def uniformity_metric(coords: np.ndarray, k: int = 4) -> float:
    
    # Fit NearestNeighbors model
    nearest_neighbors = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    
    # Get the k + 1 nearest neighbors (the first distance is to the point itself, so we ignore it)
    distances, _ = nearest_neighbors.kneighbors(coords)
    
    # Remove the first column (self-distance)
    distances = distances[:, 1:]

    # Calculate mean distance for each point to its k nearest neighbors
    mean_distances = distances.mean(axis=1)

    # Calculate the variance of the mean distances
    variance = np.var(mean_distances)

    # Uniformity score: inversely related to variance
    score = 1 / (1 + variance)

    return score
