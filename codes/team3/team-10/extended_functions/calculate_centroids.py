import numpy as np


def calculate_centroids(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    centroids: list[np.ndarray] = []
    intensities: list[float] = []

    temp_mask: np.ndarray = mask.copy()
    labels: np.ndarray = np.unique(temp_mask)

    for unique_val in labels:
        
        # Get coordinates of pixels that belong to the current region/label
        label_indices: np.ndarray = np.argwhere(temp_mask == unique_val)

        # Calculate centroids: take the mean of the label's coordinates
        centroid: np.ndarray = np.array(np.uint8(np.mean(label_indices, axis=0)))
        centroids.append(centroid)

        # Calculate the intensity of each grain/region
        boolean_mask: np.ndarray = (temp_mask == unique_val)
        intensity: float = np.mean(img[boolean_mask])
        intensities.append(intensity)
        
    # Convert centroids and intensities into numpy arrays
    centroids: np.ndarray = np.array(centroids)
    intensities: np.ndarray = np.array(intensities)

    # Normalize intensities based on mask size
    mask_size: int = temp_mask.shape[0]  # Assuming square-like or rectangular mask
    scaled_intensities: np.ndarray = mask_size * (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))

    # Extract x and y coordinates
    x: np.ndarray = centroids[:, 0]
    y: np.ndarray = centroids[:, 1]
    coordinates: np.ndarray = np.array([x, y, scaled_intensities]).T

    return coordinates
