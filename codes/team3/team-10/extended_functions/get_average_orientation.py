#function to find the average orientation of the grain
import numpy as np
import matplotlib.pyplot as plt

def get_avg_orientation(mask : np.ndarray) -> float:
    slopes_list = []

    #mask value 0 is just the background
    num_grains = len(np.unique(mask)) -1

    for i in range(1, num_grains+1):
        grain_coords = []

        indexes = np.where(mask == i)
        indexes_y = indexes[0]
        indexes_x = indexes[1]

        m, b = np.polyfit(indexes_x, indexes_y, 1)

        slopes_list.append(m)

    slopes_array = np.array(slopes_list)

    return np.average(slopes_array)

