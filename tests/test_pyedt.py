import pathlib

import numpy as np

from PyEDT.interface import *

reference = np.fromfile(pathlib.Path(__file__).parent/"20_edt.raw", dtype=np.uint32).reshape(20, 20, 20)
EDGE_SIZE = 20

def test_edt_gpu():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    assert(np.all(edt_gpu(array) == reference))

def test_edt_gpu_split():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    assert(np.all(edt_gpu_split(array, 2) == reference))
    
def test_edt_cpu():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    assert(np.all(edt_cpu(array) == reference))
    
def test_edt():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    assert(np.all(edt(array) == reference))
    
def test_benchmark_pass():
    r = run_benchmark(size_override=(20,30))
    print(r)
    assert(type(r) == dict)
    
# def test_benchmark_fail():
    # r = run_benchmark(size_override=(2500,))
    # print(r)
    # assert(type(r) == dict)