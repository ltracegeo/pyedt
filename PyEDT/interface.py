import math

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
    

def edt(A, force_method=None, minimum_segments=None):
    if force_method == None:
        method = _auto_decide_method(A)
    elif force_method in ('cpu', 'gpu', 'gpu-split'):
        method = force_method
    else:
        raise ValueError(f"force_method must be one of 'cpu', 'gpu' or 'gpu-split', was {force_method}")
        
    if method == "cpu":
        function = edt_cpu
    elif method == "gpu":
        function = edt_gpu
    elif method == "gpu-split":
        free_memory, total_memory = cuda.current_context().get_memory_info()
        expected_memory_use = (A.size*BYTES_PER_PIXEL + BASE_DEVICE_MEMORY_USE) * MEMORY_TOLERANCE_MARGIN
        segments = math.ceil(math.sqrt(expected_memory_use/free_memory))
        if minimum_segments:
            segments = max(segments, minimum_segments)
        function = lambda x: edt_gpu_split(x, segments)

    return function(A)

def run_benchmark():
    pass


def _auto_decide_method(A):
    if not cuda.is_available():
        return "cpu"
        
    free_memory, total_memory = cuda.current_context().get_memory_info()
    expected_memory_use = (A.size*BYTES_PER_PIXEL + BASE_DEVICE_MEMORY_USE) * MEMORY_TOLERANCE_MARGIN
    
    if free_memory <= expected_memory_use:
        return "gpu"
        
    return "gpu-split"
