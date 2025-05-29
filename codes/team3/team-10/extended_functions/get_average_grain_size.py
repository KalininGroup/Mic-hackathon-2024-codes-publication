#function will return average grain size in pixels
import numpy as np

def get_avgerage_grain_size(mask : np.ndarray) -> float:
    grain_sizes = []

    #mask value 0 is just the background
    num_grains = len(np.unique(mask)) -1

    for i in range(1, num_grains+1):
        indexes = np.where(mask == i)
        num_pixels = len(indexes[0])

        grain_sizes.append(num_pixels)

    return np.average(np.array(grain_sizes))