import Pyro5.api
import numpy as np
from numpy.typing import NDArray

URI = "PYRO:microscope.server@localhost:9092"
MIC_SERVER = Pyro5.api.Proxy(URI)

def _loop_raster(
        server:MIC_SERVER,
        channels:list[str],
        speedup:int,
) -> tuple[NDArray, NDArray]:
    """returns server data at tunable resolutions

    Faithfully recreates the effect of rushing a microscope.
    """
    y_coords = np.linspace(
        0, # 256 is just what's on the side of our imshow plots
        256, # generalize to mic_server.y_min and mic_server.y_max
        256 # ideally set step size to full image resolution
    )

    image = np.zeros(shape=(256, 256, 1)) # assuming single channel scans
    mask = np.zeros(new_image.shape[:-1])

    for ind, coord in enumerate(y_coords):
        #Scan every 4th line
        if ind%speedup==0:
            line = server.scan_individual_line(
                'horizontal',
                coord = coord,
                channels = channels
            )
            # TODO: dynamically allocate dims for multichannel scans
            image[ind,:,0] = line[0][0]

    return image


def scan_with_attention(
        server:MIC_SERVER,
        channels:list[str],
        speedup:int = 5,
        bbox:tuple[int,int,int,int] = None,
        fullscan:NDarray = None,
) -> tuple[NDArray, NDArray, tuple|None]:
    """Perform raster scan with increasing fidelity (raster density)
    as :param:`speedup` decreases.

    Use :param:`bbox`=(xmin, xmax, ymin, ymax) to constrain the scan
    region to a rectangular subset of the sample surface.

    Returns synchronous images of variable shape and "fidelity".
    """
    if not fullscan:
        fullscan = _loop_raster(server, channels, 10)
    cropscan = _loop_raster(server, channels, speedup)

    if bbox:
        cropscan = cropscan[bbox[0]:bbox[2], bbox[1]:bbox[3], :]

    return fullscan, cropscan, bbox


def stitch_scans(
        whole:NDarray,
        part:NDarray,
        x:int = 0,
        y:int = 0
) -> NDarray:
    """Assemble multiple scans into a single composite image.

    untested. should just inject :param:`part` into :param:`whole`

    The bbox could be indexed to get :param:`x` and :param:`y`
    """

    whole[x:x+part.shape[0], y:y+part.shape[1], ...] = part

    return whole
