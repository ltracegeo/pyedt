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
two_d_result = np.fromfile(pathlib.Path(__file__).parent/"2d_result.raw", dtype=np.uint16).reshape(40, 50, 1).astype(np.uint32)
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

def save_arrays(result, reference):    
    result.astype(np.uint16).tofile("result.raw")
    reference.astype(np.uint16).tofile("reference.raw")
    
def save_many(arrays_dict):
    for name, array in arrays_dict.items():
        array.astype(np.uint16).tofile(f'{name}.raw')
    
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

# Test 2D
def test_edt_gpu_2d():
    array = np.ascontiguousarray(test_image.copy()[: , :, test_image.shape[2]//2])
    result = edt_gpu(array)
    save_arrays(reference=two_d_result, result=result)
    assert(np.allclose(result, two_d_result[...,0], TOLERANCE))

def test_edt_gpu_split_2d():
    array = np.ascontiguousarray(test_image.copy()[: , :, test_image.shape[2]//2])
    result = edt_gpu_split(array, 2)
    save_arrays(reference=two_d_result, result=result)
    assert(np.allclose(result, two_d_result[...,0], TOLERANCE))
    
def test_edt_cpu_2d():
    array = np.ascontiguousarray(test_image.copy()[: , :, test_image.shape[2]//2])
    result = edt_cpu(array)
    save_arrays(reference=two_d_result, result=result)
    assert(np.allclose(result, two_d_result[...,0], TOLERANCE))

# Other tests
def test_edt():
    array = test_image.copy()
    result = edt(array, sqrt_result=True)
    assert(np.allclose(result, sqrt_result, TOLERANCE))
    
    
def test_benchmark_pass():
    r = run_benchmark(size_override=(10,20), test_sqrt=True)
    print(r)
    assert(type(r) == dict)


# Large image tests
rng = np.random.default_rng(42)
A = rng.binomial(1, 0.99, (200,200,200))
x, y, z = A.shape
for i in range(1, 50):
    x0, y0, z0, dx, dy, dz = rng.random(size=6)
    x0 = int(x0*x)
    x1 = int((x-x0) * dx + x0)
    y0 = int(y0*y)
    y1 = int((y-y0) * dy + y0)
    z0 = int(z0*z)
    z1 = int((z-z0) * dz + z0)
    sub_A = A[x0:x1, y0:y1, z0:z1]
    sub_A[sub_A >= 1] = i
    
A = A.astype('uint32')
B = np.ones((200,200,200), dtype = np.uint32)
B[99:101, 99:101, 99:101] = 0
x, y, z = B.shape
for i in range(1, 50):
    x0, y0, z0, dx, dy, dz = rng.random(size=6)
    x0 = int(x0*x)
    x1 = int((x-x0) * dx + x0)
    y0 = int(y0*y)
    y1 = int((y-y0) * dy + y0)
    z0 = int(z0*z)
    z1 = int((z-z0) * dz + z0)
    sub_B = B[x0:x1, y0:y1, z0:z1]
    sub_B[sub_B >= 1] = i

def test_random():
    result_gpu = edt(A, force_method='gpu')
    result_cpu = edt(A, force_method='cpu')
    result_gpu_split = edt(A, force_method='gpu-split', minimum_segments=3)
    save_many({'gpu': result_gpu, 'cpu': result_cpu, 'gpu-split': result_gpu_split})
    assert(np.allclose(result_gpu, result_cpu, TOLERANCE) and
           np.allclose(result_gpu, result_gpu_split, TOLERANCE))

def test_square():
    result_gpu = edt(B, force_method='gpu')
    result_cpu = edt(B, force_method='cpu')
    result_gpu_split = edt(B, force_method='gpu-split', minimum_segments=3)
    save_many({'gpu': result_gpu, 'cpu': result_cpu, 'gpu-split': result_gpu_split})
    assert(np.allclose(result_gpu, result_cpu, TOLERANCE) and
           np.allclose(result_gpu, result_gpu_split, TOLERANCE))
    
def test_random_sqrt():    
    result_gpu = edt(A, force_method='gpu', sqrt_result=True)
    result_cpu = edt(A, force_method='cpu', sqrt_result=True)
    result_gpu_split = edt(A, force_method='gpu-split', minimum_segments=3, sqrt_result=True)
    save_many({'gpu': result_gpu, 'cpu': result_cpu, 'gpu-split': result_gpu_split})
    assert(np.allclose(result_gpu, result_cpu, TOLERANCE) and
           np.allclose(result_gpu, result_gpu_split, TOLERANCE))
           
def test_square_sqrt():    
    result_gpu = edt(B, force_method='gpu', sqrt_result=True)
    result_cpu = edt(B, force_method='cpu', sqrt_result=True)
    result_gpu_split = edt(B, force_method='gpu-split', minimum_segments=3, sqrt_result=True)
    save_many({'gpu': result_gpu, 'cpu': result_cpu, 'gpu-split': result_gpu_split})
    assert(np.allclose(result_gpu, result_cpu, TOLERANCE) and
           np.allclose(result_gpu, result_gpu_split, TOLERANCE))

def test_random_multilabel():    
    result_gpu = edt(A, force_method='gpu', multilabel=True)
    result_cpu = edt(A, force_method='cpu', multilabel=True)
    result_gpu_split = edt(A, force_method='gpu-split', minimum_segments=3, multilabel=True)
    save_many({'gpu': result_gpu, 'cpu': result_cpu, 'gpu-split': result_gpu_split})
    assert(np.allclose(result_gpu, result_cpu, TOLERANCE) and
           np.allclose(result_gpu, result_gpu_split, TOLERANCE))
           
def test_square_multilabel():    
    result_gpu = edt(B, force_method='gpu', multilabel=True)
    result_cpu = edt(B, force_method='cpu', multilabel=True)
    result_gpu_split = edt(B, force_method='gpu-split', minimum_segments=3, multilabel=True)
    save_many({'gpu': result_gpu, 'cpu': result_cpu, 'gpu-split': result_gpu_split})
    assert(np.allclose(result_gpu, result_cpu, TOLERANCE) and
           np.allclose(result_gpu, result_gpu_split, TOLERANCE))

def test_random_border():    
    result_gpu = edt(A, force_method='gpu', closed_border=True)
    result_cpu = edt(A, force_method='cpu', closed_border=True)
    result_gpu_split = edt(A, force_method='gpu-split', minimum_segments=3, closed_border=True)
    save_many({'gpu': result_gpu, 'cpu': result_cpu, 'gpu-split': result_gpu_split})
    assert(np.allclose(result_gpu, result_cpu, TOLERANCE) and
           np.allclose(result_gpu, result_gpu_split, TOLERANCE))
           
def test_square_border():    
    result_gpu = edt(B, force_method='gpu', closed_border=True)
    result_cpu = edt(B, force_method='cpu', closed_border=True)
    result_gpu_split = edt(B, force_method='gpu-split', minimum_segments=3, closed_border=True)
    save_many({'gpu': result_gpu, 'cpu': result_cpu, 'gpu-split': result_gpu_split})
    assert(np.allclose(result_gpu, result_cpu, TOLERANCE) and
           np.allclose(result_gpu, result_gpu_split, TOLERANCE))

def test_random_scale():    
    result_gpu = edt(A, force_method='gpu', scale=(1.2, 2.4, 3.6))
    result_cpu = edt(A, force_method='cpu', scale=(1.2, 2.4, 3.6))
    result_gpu_split = edt(A, force_method='gpu-split', minimum_segments=3, scale=(1.2, 2.4, 3.6))
    save_many({'gpu': result_gpu, 'cpu': result_cpu, 'gpu-split': result_gpu_split})
    assert(np.allclose(result_gpu, result_cpu, TOLERANCE) and
           np.allclose(result_gpu, result_gpu_split, TOLERANCE))
           
def test_square_scale():    
    result_gpu = edt(B, force_method='gpu', scale=(1.2, 2.4, 3.6))
    result_cpu = edt(B, force_method='cpu', scale=(1.2, 2.4, 3.6))
    result_gpu_split = edt(B, force_method='gpu-split', minimum_segments=3, scale=(1.2, 2.4, 3.6))
    save_many({'gpu': result_gpu, 'cpu': result_cpu, 'gpu-split': result_gpu_split})
    assert(np.allclose(result_gpu, result_cpu, TOLERANCE) and
           np.allclose(result_gpu, result_gpu_split, TOLERANCE))

