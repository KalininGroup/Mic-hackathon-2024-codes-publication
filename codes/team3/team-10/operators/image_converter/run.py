import io
import pathlib
from typing import Any

import matplotlib.cm as cm
import numpy as np
from core.logger import get_logger
from core.models.messages import BytesMessage, MessageHeader, MessageSubject
from operators.operator import operator
from PIL import Image, ImageEnhance
from pydantic import BaseModel, ConfigDict, ValidationError

logger = get_logger()


skip_counter = 0
average_counter = 0
accumulated_dense = None


class OtherMetadata(BaseModel):
    model_config: ConfigDict = ConfigDict(extra="ignore")
    path: pathlib.Path
    shape: tuple[int, int]
    dtype: str


class DatasetInfo(BaseModel):
    channels: list[str] = []
    signals: list[str] = []
    units: list[str] = []
    scans: list[int] = []
    spectra: list[Any] = []
    point_clouds: list[Any] = []


@operator
def image_converter(
    inputs: BytesMessage | None, parameters: dict[str, Any]
) -> BytesMessage | None:
    global skip_counter, average_counter, accumulated_dense

    brightness = float(parameters.get("brightness", 1.0))
    contrast = float(parameters.get("contrast", 1.0))
    colormap = parameters.get("colormap", "viridis")

    if not inputs:
        return None

    try:
        other_metadata = OtherMetadata(**inputs.header.meta)
    except ValidationError:
        logger.error("Invalid message")
        return None

    data = np.frombuffer(inputs.data, dtype=other_metadata.dtype).reshape(
        other_metadata.shape
    )

    # Proceed to process the processed data
    # Normalize the processed_dense to range [0, 1]
    if data.max() > 0:
        normalized = data / data.max()
    else:
        normalized = data

    # Apply viridis colormap
    colormap = cm.get_cmap(colormap)
    colored_image = colormap(normalized)

    # Convert to 8-bit RGB
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

    # Create an RGB image
    image = Image.fromarray(colored_image, mode="RGB")

    # Apply contrast adjustment
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(contrast)

    # Apply brightness adjustment
    brightness_enhancer = ImageEnhance.Brightness(image)
    image = brightness_enhancer.enhance(brightness)

    # Save the image to a bytes buffer
    byte_array = io.BytesIO()
    image.save(byte_array, format="JPEG")
    byte_array.seek(0)
    header = MessageHeader(subject=MessageSubject.BYTES, meta={})

    # Reset accumulators for the next set
    average_counter = 0
    accumulated_dense = None

    return BytesMessage(header=header, data=byte_array.getvalue())
