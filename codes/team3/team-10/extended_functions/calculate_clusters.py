from collections import defaultdict
from typing import List, Tuple

import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def calculate_clusters(coordinates: np.ndarray, max_eps: float = 0.5, min_samples: int = 6, num_clusters: int = 5) -> np.ndarray:
    
    # Fit and transform data using StandardScaler
    scaler: StandardScaler = StandardScaler()
    x_data: np.ndarray = scaler.fit_transform(coordinates)
    
    # Apply OPTICS clustering
    optics: OPTICS = OPTICS(max_eps=max_eps, min_samples=min_samples)
    O_pred: np.ndarray = optics.fit_predict(x_data)  # Cluster labels
    
    # Organize points by cluster labels (excluding noise -1)
    clusters: defaultdict[int, List[np.ndarray]] = defaultdict(list)
    for idx, label in enumerate(optics.labels_):
        if label != -1:  # Skip noise (-1)
            clusters[label].append(x_data[idx])
    
    # Calculate centroids for each cluster
    cluster_centroids: List[np.ndarray] = []
    for label, points in clusters.items():
        centroid: np.ndarray = np.mean(np.array(points), axis=0)
        cluster_centroids.append(centroid)

    cluster_centroids: np.ndarray = np.array(cluster_centroids)
    inter_cluster_distances: np.ndarray = pairwise_distances(cluster_centroids)

    # Sort centroids based on mean of their pairwise distances
    sorted_centroids: np.ndarray = np.argsort(inter_cluster_distances.mean(axis=1))
    
    # Select the top 'num_clusters' clusters
    selected_clusters: np.ndarray = sorted_centroids[:num_clusters]

    # Calculate new centroids for the selected clusters
    updated_centroids: List[np.ndarray] = []
    for label in selected_clusters:
        selected_points: List[np.ndarray] = [x_data[idx] for idx, lbl in enumerate(optics.labels_) if lbl == label]
        if selected_points:
            centroid: np.ndarray = np.mean(np.array(selected_points), axis=0)
            updated_centroids.append(centroid)

    updated_centroids: np.ndarray = np.array(updated_centroids)

    return updated_centroids