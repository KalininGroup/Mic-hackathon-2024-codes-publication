import pathlib
from enum import Enum
from typing import Any, Callable

import numpy as np
from core.models.messages import BytesMessage
from pydantic import BaseModel, ConfigDict
from scipy import signal
from scipy.ndimage import center_of_mass

from operators.operator import operator


class GrainSizeSetting(str, Enum):
    HIGHEST = "highest"
    LOWEST = "lowest"


def select_samples_grain_size(
    mask: np.ndarray, parameters: dict[str, Any]
) -> list[list[int]]:
    no_of_bins = int(parameters.get("num_bins", 10))

    num_samples = int(parameters.get("num_samples", 1))
    setting = parameters.get("setting", GrainSizeSetting.HIGHEST)

    # Find the sizes of the different particles
    sizes = np.zeros(int(np.max(mask) + 1))

    for bin_num in range(np.max(mask) + 1):
        sizes[bin_num] = np.count_nonzero(mask == bin_num)

    # Remove the zeroth element (the background)
    sizes_new = sizes[1:]

    # Bin these clusters by size
    hist = np.histogram(sizes_new, bins=no_of_bins)
    bin_boundaries = hist[1]

    # Find the corresponding indices
    size_args: list[list[int]] = []

    for bin_num in range(no_of_bins):
        bounds = (bin_boundaries[bin_num], bin_boundaries[bin_num + 1])
        bool_arr = np.array([sizes_new >= bounds[0], sizes_new < bounds[1]])
        indices = list(np.argwhere(bool_arr.all(0)).T[0])
        size_args.append(indices)

    # Take samples from each bin
    sample_list: list[list[int]] = []

    for bin_num in range(no_of_bins):
        if len(size_args[bin_num]) > num_samples:
            samples = list(
                np.random.choice(size_args[bin_num], size=num_samples, replace=False)
            )
            sample_list.append(samples)
        else:
            sample_list.append(size_args[bin_num])

    # Find the pixel coordinates of these samples
    coords = []

    if setting == GrainSizeSetting.HIGHEST:
        bin_num = 0
    elif setting == GrainSizeSetting.LOWEST:
        bin_num = no_of_bins - 1

    sl = sample_list[bin_num]

    coord_ii = []
    for sample_num in range(len(sl)):
        mask_ii = mask == (bin_num + 1)
        coord = center_of_mass(mask_ii)
        coord_ii.append(coord)

    coords.append(coord_ii)
    return coords


# Common edge detection method for grayscale images (perfect for this)
def find_high_boundary_areas(
    mask: np.ndarray, parameters: dict[str, Any]
) -> list[list[int]]:
    print("Function: find_high_boundary_areas")

    # Convolve kernel with edges to find regions with the highest density of boundaries
    ks = 10  # Kernel size
    kernel2d = np.ones((ks, ks))
    print("Kernel size:", ks)

    boundary_density = signal.convolve2d(mask, kernel2d, mode="same", boundary="fill")

    # Generate scan target coordinates
    scan_size = 10
    num_targets = int(parameters.get("num_targets", 10))

    target_map = boundary_density.copy()
    coords: list[list[int]] = []

    for _ in range(num_targets):
        # Find maximum density of boundaries in the current target map
        coord = np.unravel_index(np.argmax(target_map), target_map.shape)

        # Calculate scan boundaries
        x_min = np.max((int(coord[1] - scan_size / 2), 0))
        x_max = np.min((int(coord[1] + scan_size / 2), target_map.shape[1]))
        y_min = np.max((int(coord[0] - scan_size / 2), 0))
        y_max = np.min((int(coord[0] + scan_size / 2), target_map.shape[0]))

        # Block out an area of scan_size**2 around to stop repeat imaging of the same region
        target_map[y_min:y_max, x_min:x_max] = 0

        # Append the found coordinates as a list
        coords.append([int(coord[0]), int(coord[1])])

    return coords


class OtherMetadata(BaseModel):
    model_config: ConfigDict = ConfigDict(extra="ignore")
    path: pathlib.Path
    shape: tuple[int, ...]
    dtype: str


class MethodSelection(str, Enum):
    GRAIN_SIZE = "grain_size"
    GRAIN_BOUNDARIES = "grain_boundaries"


MethodCallableType = Callable[[np.ndarray, dict[str, Any]], list[list[int]]]


@operator
def mask_operation(
    inputs: BytesMessage | None, parameters: dict[str, Any]
) -> BytesMessage | None:
    if inputs is None:
        return None

    other_metadata = OtherMetadata(**inputs.header.meta)

    mask = np.frombuffer(inputs.data, dtype=other_metadata.dtype).reshape(
        other_metadata.shape
    )

    param_method: MethodSelection = parameters.get(
        "mask_function", MethodSelection.GRAIN_BOUNDARIES
    )

    method_map: dict[MethodSelection, MethodCallableType] = {
        MethodSelection.GRAIN_SIZE: select_samples_grain_size,
        MethodSelection.GRAIN_BOUNDARIES: find_high_boundary_areas,
    }
    method = method_map[param_method]

    try:
        data = method(mask, parameters)
    except Exception:
        raise  # Reraise the error for further handling if needed

    data_np = np.array(data)

    # convert to 2D array
    data_np = data_np[0]

    meta = OtherMetadata(
        path=other_metadata.path,
        shape=data_np.shape,
        dtype=str(data_np.dtype),
    )
    header = inputs.header
    header.meta.update(meta.model_dump())

    return BytesMessage(header=header, data=data_np.tobytes())
