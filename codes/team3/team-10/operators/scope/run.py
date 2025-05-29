import multiprocessing
import pathlib
import time
from typing import Any, cast

import DTMicroscope.server.server_afm
import numpy as np
import Pyro5.api
import Pyro5.errors
from DTMicroscope.server.server_afm import main_server
from pydantic import BaseModel, ConfigDict

from core.models.messages import BytesMessage, MessageHeader, MessageSubject
from operators.operator import DATA_DIRECTORY, dependencies, operator


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


URI = "PYRO:microscope.server@localhost:9092"
MIC_SERVER = Pyro5.api.Proxy(URI)
MIC_SERVER = cast(DTMicroscope.server.server_afm.MicroscopeServer, MIC_SERVER)
the_file = pathlib.Path(DATA_DIRECTORY) / "data.h5"
the_dataset_info: DatasetInfo = DatasetInfo()
the_data = np.random.rand(10, 10)


@dependencies
def deps():
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


sent_first = False


@operator
def afm_microscope(
    inputs: BytesMessage | None, parameters: dict[str, Any]
) -> BytesMessage | None:
    global MIC_SERVER, the_dataset_info, the_data, sent_first
    meta1 = the_dataset_info.model_dump()
    meta2 = OtherMetadata(
        path=the_file, shape=the_data.shape, dtype=str(the_data.dtype)
    ).model_dump()
    meta = {**meta1, **meta2}
    header = MessageHeader(subject=MessageSubject.BYTES, meta=meta)
    if sent_first:
        time.sleep(60)
        print("Sending image")
    else:
        print("Sending first image")
        sent_first = True
    return BytesMessage(header=header, data=the_data.tobytes())
