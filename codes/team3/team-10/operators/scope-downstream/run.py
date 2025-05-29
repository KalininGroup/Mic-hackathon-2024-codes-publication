import multiprocessing
import pathlib
import time
from typing import Any, cast

import DTMicroscope.server.server_afm
import numpy as np
import Pyro5.api
import Pyro5.errors
from core.models.messages import BytesMessage, MessageHeader, MessageSubject
from operators.operator import DATA_DIRECTORY, dependencies, operator
from pydantic import BaseModel, ConfigDict


class OtherMetadata(BaseModel):
    model_config: ConfigDict = ConfigDict(extra="ignore")
    path: pathlib.Path
    shape: tuple[int, ...]
    dtype: str


class DatasetInfo(BaseModel):
    channels: list[str] = []
    signals: list[str] = []
    units: list[str] = []
    scans: list[int] = []
    spectra: list[Any] = []
    point_clouds: list[Any] = []

    @classmethod
    def from_info(cls, info):
        info_dict = {item[0]: item[1] for item in info}

        try:
            return cls(
                channels=info_dict["channels"],
                signals=info_dict["signals"],
                units=info_dict["units"],
                scans=info_dict["scans"],
                spectra=info_dict["spectra"],
                point_clouds=info_dict["point_clouds"],
            )
        except KeyError as e:
            raise ValueError(f"Missing key in info: {e}")


URI = "PYRO:microscope.server@localhost:9091"
MIC_SERVER = Pyro5.api.Proxy(URI)
MIC_SERVER = cast(DTMicroscope.server.server_afm.MicroscopeServer, MIC_SERVER)
the_file = pathlib.Path(DATA_DIRECTORY) / "data.h5"
if not the_file.exists():
    print(f"File {the_file} does not exist")
the_dataset_info: DatasetInfo = DatasetInfo()
the_data = np.random.rand(10, 10)


@dependencies
def deps():
    # This is a cluge: we already have a server running in the other operator.
    # We previously tried to connect to it through this operator, but got the following error:
    # Error in kernel: the calling thread is not the owner of this proxy, create a new proxy in this thread or transfer ownership.
    # So here, we make a new server in this operator and connect to it.
    def main_server():
        host = "0.0.0.0"
        daemon = Pyro5.api.Daemon(port=9091)
        uri = daemon.register(
            DTMicroscope.server.server_afm.MicroscopeServer,
            objectId="microscope.server",
        )
        print("Server is ready. Object uri =", uri)
        daemon.requestLoop()

    process = multiprocessing.Process(target=main_server)
    process.start()
    global MIC_SERVER, the_data, the_dataset_info
    retries = 0
    max_retries = 5
    while True:
        try:
            MIC_SERVER.initialize_microscope("AFM", data_path=str(the_file))
            MIC_SERVER.setup_microscope()
            the_dataset_info = DatasetInfo.from_info(MIC_SERVER.get_dataset_info())

            data = MIC_SERVER.get_scan(channels=["HeightRetrace"])
            if data is None:
                raise ValueError("No data returned from get_scan")
            array_list, shape, dtype = data
            the_data = np.array(array_list, dtype=dtype).reshape(shape)
            the_data = the_data[0]
            print("INITIALIZED MICROSCOPE")
            break
        except Pyro5.errors.CommunicationError:
            retries += 1
            time.sleep(1)
            if retries > max_retries:
                raise
    yield
    process.terminate()
    process.join()


def get_data_around_coordinate(
    arr: np.ndarray, coordinate: tuple[int, int], size: int = 20
) -> np.ndarray:
    y, x = coordinate
    y = int(y)
    x = int(x)

    # Determine the boundaries of the sub-array
    y_min = max(y - size, 0)
    y_max = min(y + size, arr.shape[0])  # Height of the array
    x_min = max(x - size, 0)
    x_max = min(x + size, arr.shape[1])  # Width of the array

    # Extract the data around the coordinate
    return arr[y_min:y_max, x_min:x_max]


@operator
def afm_microscope(
    inputs: BytesMessage | None, parameters: dict[str, Any]
) -> BytesMessage | None:
    global MIC_SERVER, the_dataset_info, the_data
    if inputs is None:
        return None

    other_metadata = OtherMetadata(**inputs.header.meta)
    coords = np.frombuffer(inputs.data, dtype=other_metadata.dtype).reshape(
        other_metadata.shape
    )

    print(coords)

    if coords.ndim != 2:
        raise ValueError("Coordinates must be 2D")

    if coords.shape[1] != 2:
        raise ValueError("Coordinates must have 2 columns")

    the_place_to_go_to = coords[0]

    # Coordinates should be in the form (y, x)
    print("Going to coordinates", the_place_to_go_to)
    # server.go_to(the_place_to_go_to[1], the_place_to_go_to[0])

    # In the future, we will use code from daq-agent (see operators/scope) to
    # extract a scan
    print("Getting scan")
    # data = server.get_scan(channels=["HeightRetrace"])

    data = the_data.copy()
    cropped_data = get_data_around_coordinate(data, the_place_to_go_to)

    meta1 = the_dataset_info.model_dump()
    meta2 = OtherMetadata(
        path=the_file, shape=cropped_data.shape, dtype=str(cropped_data.dtype)
    ).model_dump()
    meta = {**meta1, **meta2}
    header = MessageHeader(subject=MessageSubject.BYTES, meta=meta)
    return BytesMessage(header=header, data=cropped_data.tobytes())
