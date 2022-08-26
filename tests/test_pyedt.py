import pathlib

import numpy as np
from scipy import ndimage

from pyedt.interface import *

test_image = np.fromfile(pathlib.Path(__file__).parent/"test_array.raw", dtype=np.uint16).reshape(40, 50, 60).astype(np.uint32)
border_result = np.fromfile(pathlib.Path(__file__).parent/"border_result.raw", dtype=np.uint16).reshape(40, 50, 60).astype(np.uint32)
multilabel_result = np.fromfile(pathlib.Path(__file__).parent/"multilabel_result.raw", dtype=np.uint16).reshape(40, 50, 60).astype(np.uint32)
scale_result = np.fromfile(pathlib.Path(__file__).parent/"scale_result_f32.raw", dtype=np.float32).reshape(40, 50, 60)
simple_result = np.fromfile(pathlib.Path(__file__).parent/"simple_result.raw", dtype=np.uint16).reshape(40, 50, 60).astype(np.uint32)
sqrt_result = np.fromfile(pathlib.Path(__file__).parent/"sqrt_result_f32.raw", dtype=np.float32).reshape(40, 50, 60)
EDGE_SIZE = (40, 50, 60)
SAVE_IMAGE = True
TOLERANCE = 1e-3

def array_generator():
    array = np.zeros(EDGE_SIZE, dtype=np.uint32)
    array[EDGE_SIZE[0]//4:3*EDGE_SIZE[0]//4,
          EDGE_SIZE[1]//4:3*EDGE_SIZE[1]//4,
          EDGE_SIZE[2]//4:3*EDGE_SIZE[2]//4] = 1
    array[EDGE_SIZE[0]//2:4*EDGE_SIZE[0]//5,
          EDGE_SIZE[1]//2:4*EDGE_SIZE[1]//5,
          EDGE_SIZE[2]//2:4*EDGE_SIZE[2]//5] = 2
    array[3*EDGE_SIZE[0]//8:5*EDGE_SIZE[0]//8,
          3*EDGE_SIZE[1]//8:5*EDGE_SIZE[1]//8,
          3*EDGE_SIZE[2]//8:5*EDGE_SIZE[2]//8] = 3
    array[EDGE_SIZE[0]//4:3*EDGE_SIZE[0]//4,
          3*EDGE_SIZE[1]//4:7*EDGE_SIZE[1]//8,
          EDGE_SIZE[2]//4:3*EDGE_SIZE[2]//4] = 5
    array[2*EDGE_SIZE[0]//5:3*EDGE_SIZE[0]//5,
          3*EDGE_SIZE[1]//4:7*EDGE_SIZE[1]//8,
          0:1*EDGE_SIZE[2]//3] = 7
    return array
    
# GPU test
def test_edt_gpu():
    array = test_image.copy()
    result = edt_gpu(array)
    assert(np.allclose(result, simple_result, TOLERANCE))   
    
def test_edt_gpu_scaled():
    array = test_image.copy()
    result = edt_gpu(array, scale=(1.2, 2.4, 3.6))
    assert(np.allclose(result, scale_result, TOLERANCE))
    
def test_edt_gpu_multilabel():
    array = test_image.copy()
    result = edt_gpu(array, multilabel=True)
    assert(np.allclose(result, multilabel_result, TOLERANCE))

def test_edt_gpu_border():
    array = test_image.copy()
    result = edt_gpu(array, closed_border=True)
    assert(np.allclose(result, border_result, TOLERANCE))
    
def test_edt_gpu_sqrt():
    array = test_image.copy()
    result = edt_gpu(array, sqrt_result=True)
    assert(np.allclose(result, sqrt_result, TOLERANCE))

# GPU split test
def test_edt_split_gpu():
    array = test_image.copy()
    result = edt_gpu_split(array, 2)
    assert(np.allclose(result, simple_result, TOLERANCE))   
    
def test_edt_split_gpu_scaled():
    array = test_image.copy()
    result = edt_gpu_split(array, 2, scale=(1.2, 2.4, 3.6))
    assert(np.allclose(result, scale_result, TOLERANCE))
    
def test_edt_split_gpu_multilabel():
    array = test_image.copy()
    result = edt_gpu_split(array, 2, multilabel=True)
    assert(np.allclose(result, multilabel_result, TOLERANCE))

def test_edt_split_gpu_border():
    array = test_image.copy()
    result = edt_gpu_split(array, 2, closed_border=True)
    assert(np.allclose(result, border_result, TOLERANCE))
    
def test_edt_split_gpu_sqrt():
    array = test_image.copy()
    result = edt_gpu_split(array, 2, sqrt_result=True)
    assert(np.allclose(result, sqrt_result, TOLERANCE))
    
# CPU test    
def test_edt_cpu():
    array = test_image.copy()
    result = edt_cpu(array)
    assert(np.allclose(result, simple_result, TOLERANCE))   
    
def test_edt_cpu_scaled():
    array = test_image.copy()
    result = edt_cpu(array, scale=(1.2, 2.4, 3.6))
    assert(np.allclose(result, scale_result, TOLERANCE))
    
def test_edt_cpu_multilabel():
    array = test_image.copy()
    result = edt_cpu(array, multilabel=True)
    assert(np.allclose(result, multilabel_result, TOLERANCE))

def test_edt_cpu_border():
    array = test_image.copy()
    result = edt_cpu(array, closed_border=True)
    assert(np.allclose(result, border_result, TOLERANCE))
    
def test_edt_cpu_sqrt():
    array = test_image.copy()
    result = edt_cpu(array, sqrt_result=True)
    assert(np.allclose(result, sqrt_result, TOLERANCE))


# test 2D
def test_edt_gpu_2d():
    array = array_generator()[:, EDGE_SIZE//2, :]
    reference = (ndimage.distance_transform_edt(array)**2).astype(np.uint32)
    reference.astype(np.uint16).tofile("ndimage_2d.raw")
    edt_gpu(array).astype(np.uint16).tofile("gpu_2d.raw")
    assert(np.allclose(np.sqrt(reference), np.sqrt(edt_gpu(array)), atol=1.1))


def test_edt_gpu_split_2d():
    array = array_generator()[:, EDGE_SIZE//2, :]
    reference = (ndimage.distance_transform_edt(array)**2).astype(np.uint32)
    assert(np.allclose(np.sqrt(reference), np.sqrt(edt_gpu_split(array, 2)), atol=1.1))
    

def test_edt_cpu_2d():
    array = array_generator()[:, EDGE_SIZE//2, :]
    reference = (ndimage.distance_transform_edt(array)**2).astype(np.uint32)
    reference.astype(np.uint16).tofile("ndimage_2d.raw")
    assert(np.allclose(np.sqrt(reference), np.sqrt(edt_cpu(array)), atol=1.1))


def test_edt():
    array = array_generator()
    assert(np.all(edt(array) == reference))
    
    
def test_benchmark_pass():
    r = run_benchmark(size_override=(10,20), test_sqrt=True)
    print(r)
    assert(type(r) == dict)


rng = np.random.default_rng(42)
A = rng.binomial(1, 0.99, (40,40,40))
A = A.astype('uint32')
B = np.ones((50,50,50), dtype = np.uint32)
B[24:26, 24:26, 24:26] = 0

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
    
