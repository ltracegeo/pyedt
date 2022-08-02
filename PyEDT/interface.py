import math
import time

from numba import cuda, float32, uint16, uint32, njit, prange
from numba.cuda.cudadrv.driver import CudaAPIError
import numpy as np

from .functions import *


BASE_DEVICE_MEMORY_USE = 5000000
BYTES_PER_PIXEL = 4
MEMORY_TOLERANCE_MARGIN = 1.1


def edt_gpu(A):
    gpu = cuda.get_current_device()
    max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
    A_d = cuda.to_device(A)
    x_dim, y_dim, z_dim = A.shape
    for grid_dim_1, grid_dim_2, line_length, gedt_compiler in (
    (y_dim, z_dim, x_dim, compile_gedt_x),
    (x_dim, z_dim, y_dim, compile_gedt_y),
    (x_dim, y_dim, z_dim, compile_gedt_z)
    ):
        voxels_per_thread = math.ceil(line_length/max_threads_per_block)
        threads_per_block = math.ceil(line_length/voxels_per_thread)
        gedt = gedt_compiler(line_length, voxels_per_thread)
        gedt[(grid_dim_1, grid_dim_2), threads_per_block](A_d)

    B = A_d.copy_to_host()
    del A_d
    return B


def edt_gpu_split(A, segments):
    gpu = cuda.get_current_device()
    max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
    x_dim, y_dim, z_dim = A.shape
    
    for grid_axis_1, grid_axis_2, line_axis, gedt_compiler in (
    (1, 2, 0, compile_gedt_x),
    (0, 2, 1, compile_gedt_y),
    (0, 1, 2, compile_gedt_z)
    ):
        grid_dim_1 = A.shape[grid_axis_1]
        grid_dim_2 = A.shape[grid_axis_2]
        line_length = A.shape[line_axis]
        
        compiled_gedts = dict()
        voxels_per_thread = math.ceil(line_length/max_threads_per_block)
        threads_per_block = math.ceil(line_length/voxels_per_thread)
    
        slices_tuple = [(slice(None),
                         slice((i)*grid_dim_1//(segments), (i+1)*grid_dim_1//(segments)),
                         slice((j)*grid_dim_2//(segments), (j+1)*grid_dim_2//(segments)))
                         for i in range(segments) for j in range(segments)]

        for slices in slices_tuple:
            if (line_length, voxels_per_thread) not in compiled_gedts.keys():
                compiled_gedts[(line_length, voxels_per_thread)] = gedt_compiler(line_length, voxels_per_thread)
            gedt = compiled_gedts[(line_length, voxels_per_thread)]
            ordered_slices = dict()
            ordered_slices[line_axis] = slices[0]
            ordered_slices[grid_axis_1] = slices[1]
            ordered_slices[grid_axis_2] = slices[2]
            A_d = cuda.to_device(np.ascontiguousarray(A[ordered_slices[0], ordered_slices[1], ordered_slices[2]]))
            gedt[(slices[1].stop - slices[1].start, slices[2].stop - slices[2].start), threads_per_block](A_d)    
            A[ordered_slices[0], ordered_slices[1], ordered_slices[2]] = A_d.copy_to_host()

    del A_d
    return A
    
    
def edt_cpu(A):
    B = np.empty_like(A)
    single_pass_erosion_x(A, B)
    single_pass_erosion_y(B)
    single_pass_erosion_z(B)
    return B
    

def edt(A, force_method=None, minimum_segments=3):
    if force_method == None:
        method = _auto_decide_method(A)
    elif force_method in ('cpu', 'gpu', 'gpu-split'):
        method = force_method
    else:
        raise ValueError(f"force_method must be one of 'cpu', 'gpu' or 'gpu-split', was {force_method}")
        
    if method == "cpu":
        print("using cpu")
        function = edt_cpu
    elif method == "gpu":
        print("using gpu")
        function = edt_gpu
    elif method == "gpu-split":
        free_memory, total_memory = cuda.current_context().get_memory_info()
        expected_memory_use = (A.size*BYTES_PER_PIXEL + BASE_DEVICE_MEMORY_USE) * MEMORY_TOLERANCE_MARGIN
        segments = math.ceil(math.sqrt(expected_memory_use/free_memory))
        if minimum_segments:
            segments = max(segments, minimum_segments)
        print(f"using gpu {segments} segments")
        function = lambda x: edt_gpu_split(x, segments)

    return function(A)


def run_benchmark(size_override=None):
    try:
        from scipy import ndimage
    except ModuleNotFoundError as e:
        print("failed to load scipy", e)
        ndimage_loaded = False
    else:
        ndimage_loaded = True
       
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as e:
        print("failed to load SimpleITK", e)
        sitk_loaded = False
    else:
        sitk_loaded = True

    try:
        import edt as edt_ws
    except ModuleNotFoundError as e:
        print("failed to load edt", e)
        edt_loaded = False
    else:
        edt_loaded = True

    results = dict()
    size = 20
    A = np.zeros((size, size, size), dtype = np.uint32)
    A[size//4:3*size//4, size//4:3*size//4, size//4:3*size//4] = 1
    _ = edt(A, force_method='cpu')
    _ = edt(A, force_method='gpu')
    _ = edt(A, force_method='gpu-split')
        
    if size_override:
        sizes = size_override
    else:
        sizes = (200, 500, 1000, 1500, 2000)
        
    for size in sizes:
        A = np.zeros((size, size, size), dtype = np.uint32)
        A[size//4:3*size//4, size//4:3*size//4, size//4:3*size//4] = 1
    
        for method in ('cpu', 'gpu'):
            try:
                start_time = time.monotonic()
                _ = edt(A, force_method=method)
                end_time = time.monotonic()
            except Exception as e:
                print(method, size, e)
                results[f"{method}_{size}"] = "fail"
            else:
                results[f"{method}_{size}"] = end_time - start_time
        for segments in (2, 3, 4):
            try:
                start_time = time.monotonic()
                _ = edt(A, force_method="gpu-split", minimum_segments = segments)
                end_time = time.monotonic()
            except Exception as e:
                print("gpu-split", size, e)
                results[f"gpu-split_{segments}_{size}"] = "fail"
            else:
                results[f"gpu-split_{segments}_{size}"] = end_time - start_time
                
        if ndimage_loaded:
            try:
                start_time = time.monotonic()
                _ = ndimage.distance_transform_edt(A)
                end_time = time.monotonic()
            except Exception as e:
                print("ndimage", size, e)
                results[f"ndimage_{segments}_{size}"] = "fail"
            else:
                results[f"ndimage_{segments}_{size}"] = end_time - start_time
        
        if sitk_loaded:
            try:
                start_time = time.monotonic()
                _ = sitk.SignedMaurerDistanceMap(sitk.GetImageFromArray(A))
                end_time = time.monotonic()
            except Exception as e:
                print("sitk", size, e)
                results[f"sitk_{segments}_{size}"] = "fail"
            else:
                results[f"sitk_{segments}_{size}"] = end_time - start_time
            
        if edt_loaded:
            try:
                start_time = time.monotonic()
                _ = edt_ws.edt(A)
                end_time = time.monotonic()
            except Exception as e:
                print("edt", size, e)
                results[f"edt_{segments}_{size}"] = "fail"
            else:
                results[f"edt_{segments}_{size}"] = end_time - start_time
            
    return results
            
    
def _auto_decide_method(A):
    if not cuda.is_available():
        return "cpu"
        
    free_memory, total_memory = cuda.current_context().get_memory_info()
    expected_memory_use = (A.size*BYTES_PER_PIXEL + BASE_DEVICE_MEMORY_USE) * MEMORY_TOLERANCE_MARGIN
    
    if free_memory > expected_memory_use:
        return "gpu"
        
    return "gpu-split"
