import pathlib

import numpy as np
from scipy import ndimage

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

rng = np.random.default_rng(42)
A = rng.binomial(1, 0.9, (100,100,100))
A = A.astype('uint32')
# A = np.ones((50,50,50), dtype = np.uint32)
# A[24:26, 24:26, 24:26] = 0

def test_gpu_cpu_results():
    
    A_gpu = edt(A, force_method='gpu')
    A_cpu = edt(A, force_method='cpu')

    A_gpu.astype(np.uint16).tofile('A_gpu.raw')
    A_cpu.astype(np.uint16).tofile('A_cpu.raw')
    assert(np.all(A_gpu == A_cpu))
    
def test_gpu_results():
    
    A_gpu = edt(A, force_method='gpu')
    A_ndimage = (ndimage.distance_transform_edt(A)**2).astype(np.uint32)
    A_gpu.astype(np.uint16).tofile('A_gpu.raw')
    A_ndimage.astype(np.uint16).tofile('A_ndimage.raw')
    assert(np.allclose(A_ndimage, A_gpu, atol=1))

def test_cpu_results():
    
    A_cpu = edt(A, force_method='cpu')
    A_ndimage = (ndimage.distance_transform_edt(A)**2).astype(np.uint32)
    A_cpu.astype(np.uint16).tofile('A_cpu.raw')
    assert(np.allclose(A_ndimage, A_cpu, atol=1))
