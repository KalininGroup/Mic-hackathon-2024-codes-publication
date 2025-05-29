import multiprocessing
import pathlib
import time
from typing import cast

import DTMicroscope
import DTMicroscope.server.server_afm
import numpy as np
import Pyro5.api
import Pyro5.errors
import pytest
from DTMicroscope.server.server_afm import main_server

HERE = pathlib.Path(__file__).parent
DATA_DIR = (
    HERE.parent / "thirdparty" / "rama-data" / "TopoForHack" / "material1_perovksite"
)
filenames = DATA_DIR.glob("*.h5")
H5_FILES = cast(list, cast(str, [str(DATA_DIR / f) for f in filenames]))


@pytest.fixture(scope="module")
def run_server():
    process = multiprocessing.Process(target=main_server)
    process.start()
    try:
        yield
    finally:
        process.terminate()
        process.join()


def test_load_data(run_server):
    uri = "PYRO:microscope.server@localhost:9092"
    mic_server = Pyro5.api.Proxy(uri)
    mic_server = cast(DTMicroscope.server.server_afm.MicroscopeServer, mic_server)

    retries = 0
    max_retries = 5
    while True:
        try:
            mic_server.initialize_microscope("AFM", data_path=str(H5_FILES[0]))
            break
        except Pyro5.errors.CommunicationError:
            retries += 1
            time.sleep(1)
            if retries > max_retries:
                raise
    mic_server.setup_microscope()
    info = mic_server.get_dataset_info()
    print(info)
    array_list, shape, dtype = mic_server.get_scan(
        channels=["HeightRetrace"],
    )
    arr1 = np.array(array_list, dtype=dtype).reshape(shape)

    array_list, shape, dtype = mic_server.get_scan(channels=["Channel_000"])
    arr2 = np.array(array_list, dtype=dtype).reshape(shape)

    assert arr1.shape == arr2.shape
    assert arr1.dtype == arr2.dtype
    assert np.allclose(arr1, arr2)
