import pathlib

import numpy as np

from PyEDT.interface import edt_gpu, edt_cpu

reference = np.fromfile(pathlib.Path(__file__).parent/"20_edt.raw", dtype=np.uint32).reshape(20, 20, 20)
EDGE_SIZE = 20

def test_edt_gpu():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    assert(np.all(edt_gpu(array) == reference))

def test_edt_gpu_split():
    pass
    
def test_est_cpu():
    pass
