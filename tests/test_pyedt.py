import pathlib

import numpy as np
from scipy import ndimage

from pyedt.interface import *

reference = np.fromfile(pathlib.Path(__file__).parent/"20_edt.raw", dtype=np.uint32).reshape(20, 20, 20)
EDGE_SIZE = 20

def test_edt_gpu():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    assert(np.all(edt_gpu(array) == reference))
    
def test_edt_gpu_scaled():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    assert(edt_gpu(array, scale=(1.2, 2.4, 3.6)).dtype == np.float32)

def test_edt_gpu_split():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    assert(np.all(edt_gpu_split(array, 2) == reference))
    
def test_edt_gpu_split_scaled():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    assert(edt_gpu_split(array, 2, scale=(1.2, 2.4, 3.6)).dtype == np.float32)
    
def test_edt_cpu():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    assert(np.all(edt_cpu(array) == reference))
    
def test_edt_cpu_scaled():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    assert(edt_cpu(array, scale=(1.2, 2.4, 3.6)).dtype == np.float32)

def test_edt_gpu_2d():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    reference = (ndimage.distance_transform_edt(array)**2).astype(np.uint32)
    reference.astype(np.uint16).tofile("ndimage_2d.raw")
    edt_gpu(array).astype(np.uint16).tofile("gpu_2d.raw")
    assert(np.allclose(np.sqrt(reference), np.sqrt(edt_gpu(array)), atol=1.1))


def test_edt_gpu_split_2d():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    reference = (ndimage.distance_transform_edt(array)**2).astype(np.uint32)
    assert(np.allclose(np.sqrt(reference), np.sqrt(edt_gpu_split(array, 2)), atol=1.1))
    

def test_edt_cpu_2d():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    reference = (ndimage.distance_transform_edt(array)**2).astype(np.uint32)
    reference.astype(np.uint16).tofile("ndimage_2d.raw")
    # edt_cpu(array).astype(np.uint16).tofile("cpu_2d.raw")
    assert(np.allclose(np.sqrt(reference), np.sqrt(edt_cpu(array)), atol=1.1))


def test_edt():
    array = np.zeros((EDGE_SIZE, EDGE_SIZE, EDGE_SIZE), dtype=np.uint32)
    array[EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4,
          EDGE_SIZE//4:3*EDGE_SIZE//4] = 1
    assert(np.all(edt(array) == reference))
    
def test_benchmark_pass():
    r = run_benchmark(size_override=(10,20), test_sqrt=True)
    print(r)
    assert(type(r) == dict)

rng = np.random.default_rng(42)
A = rng.binomial(1, 0.99, (40,40,40))
A = A.astype('uint32')
# A = np.ones((50,50,50), dtype = np.uint32)
# A[24:26, 24:26, 24:26] = 0

def test_gpu_cpu_results():
    
    A_gpu = edt(A, force_method='gpu')
    A_cpu = edt(A, force_method='cpu')

    A_gpu.astype(np.uint16).tofile('A_gpu.raw')
    A_cpu.astype(np.uint16).tofile('A_cpu.raw')
    assert(np.allclose(np.sqrt(A_gpu), np.sqrt(A_cpu), atol=1.1))
    
def test_gpu_results():
    
    A_gpu = edt(A, force_method='gpu')
    A_ndimage = (ndimage.distance_transform_edt(A)**2).astype(np.uint32)
    #A_gpu.astype(np.uint16).tofile('A_gpu.raw')
    #A_ndimage.astype(np.uint16).tofile('A_ndimage.raw')
    assert(np.allclose(A_ndimage, A_gpu, atol=1.1))

def test_cpu_results():
    
    A_cpu = edt(A, force_method='cpu')
    A_ndimage = (ndimage.distance_transform_edt(A)**2).astype(np.uint32)
    #A_cpu.astype(np.uint16).tofile('A_cpu.raw')
    A_ndimage.astype(np.uint16).tofile('A_ndimage.raw')
    print((np.where(A_cpu > A_ndimage, A_cpu - A_ndimage, A_ndimage - A_cpu) != 0).sum())
    print((np.where(A_cpu > A_ndimage, A_cpu - A_ndimage, A_ndimage - A_cpu) >  1).sum())
    print(np.unique(np.where(A_cpu > A_ndimage, A_cpu - A_ndimage, A_ndimage - A_cpu)))
    print(np.unique(np.where(A_cpu > A_ndimage, np.sqrt(A_cpu) - np.sqrt(A_ndimage), np.sqrt(A_ndimage) - np.sqrt(A_cpu))))
    assert(np.allclose(np.sqrt(A_ndimage), np.sqrt(A_cpu), atol=1.1))
    
def test_gpu_squared_results():
    
    A_gpu = edt(A, force_method='gpu', sqrt_result=True)
    A_ndimage = ndimage.distance_transform_edt(A).astype(np.float32)
    #A_gpu.astype(np.uint16).tofile('A_gpu.raw')
    #A_ndimage.astype(np.uint16).tofile('A_ndimage.raw')
    assert(np.allclose(A_ndimage, A_gpu, atol=1.1))

def test_cpu_squared_results():
    
    A_cpu = edt(A, force_method='cpu', sqrt_result=True)
    A_ndimage = ndimage.distance_transform_edt(A).astype(np.float32)
    A_cpu.astype(np.uint16).tofile('A_cpu.raw')
    A_ndimage.astype(np.uint16).tofile('A_ndimage.raw')
    print((np.where(A_cpu > A_ndimage, A_cpu - A_ndimage, A_ndimage - A_cpu) != 0).sum())
    print((np.where(A_cpu > A_ndimage, A_cpu - A_ndimage, A_ndimage - A_cpu) >  1).sum())
    print(np.unique(np.where(A_cpu > A_ndimage, A_cpu - A_ndimage, A_ndimage - A_cpu)))
    print(np.unique(np.where(A_cpu > A_ndimage, np.sqrt(A_cpu) - np.sqrt(A_ndimage), np.sqrt(A_ndimage) - np.sqrt(A_cpu))))
    assert(np.allclose(np.sqrt(A_ndimage), np.sqrt(A_cpu), atol=1.1))
    
def test_gpu_border_results():
    
    A_gpu = edt(A, force_method='gpu', closed_border=True)
    border_A = np.zeros((A.shape[0]+2, A.shape[1]+2, A.shape[2]+2), dtype = np.uint32)
    border_A[1:-1, 1:-1, 1:-1] = A
    A_ndimage = (ndimage.distance_transform_edt(border_A)**2).astype(np.uint32)[1:-1, 1:-1, 1:-1]
    A_gpu.astype(np.uint16).tofile('A_gpu.raw')
    A_ndimage.astype(np.uint16).tofile('A_ndimage.raw')
    #A_gpu.astype(np.uint16).tofile('A_gpu.raw')
    #A_ndimage.astype(np.uint16).tofile('A_ndimage.raw')
    assert(np.allclose(np.sqrt(A_ndimage), np.sqrt(A_gpu), atol=1.1))

def test_cpu_border_results():
    
    A_cpu = edt(A, force_method='cpu', closed_border=True)
    #A_ndimage = (ndimage.distance_transform_edt(A)**2).astype(np.uint32)
    border_A = np.zeros((A.shape[0]+2, A.shape[1]+2, A.shape[2]+2), dtype = np.uint32)
    border_A[1:-1, 1:-1, 1:-1] = A
    A_ndimage = (ndimage.distance_transform_edt(border_A)**2).astype(np.uint32)[1:-1, 1:-1, 1:-1]
    A_cpu.astype(np.uint16).tofile('A_cpu.raw')
    A_ndimage.astype(np.uint16).tofile('A_ndimage.raw')
    print((np.where(A_cpu > A_ndimage, A_cpu - A_ndimage, A_ndimage - A_cpu) != 0).sum())
    print((np.where(A_cpu > A_ndimage, A_cpu - A_ndimage, A_ndimage - A_cpu) >  1).sum())
    print(np.unique(np.where(A_cpu > A_ndimage, A_cpu - A_ndimage, 0)))
    print(np.unique(np.where(A_cpu > A_ndimage, 0, A_ndimage - A_cpu)))
    print(np.unique(np.where(A_cpu > A_ndimage, np.sqrt(A_cpu) - np.sqrt(A_ndimage), np.sqrt(A_ndimage) - np.sqrt(A_cpu))))
    assert(np.allclose(np.sqrt(A_ndimage), np.sqrt(A_cpu), atol=1.1))
    
# def test_benchmark_cpu():
    # size = 200
    # A = np.zeros((size, size, size//2), dtype = np.uint32)
    # A[size//4:3*size//4, size//4:3*size//4, size//4:3*size//4] = 1
    # A_cpu = edt(A, force_method='cpu')
    # assert(True)
    
