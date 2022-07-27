import math

from numba import cuda, float32, uint16, uint32, njit, prange
from numba.cuda.cudadrv.driver import CudaAPIError
import numpy as np

from .functions import *

cuda.select_device(0)
print(cuda.gpus[0].name)
cuda.is_available()
cuda.detect()
gpu = cuda.get_current_device()
print(f"maxThreadsPerBlock: {(MAX_THREADS_PER_BLOCK := gpu.MAX_THREADS_PER_BLOCK)}")
print(f"maxBlockDimX:  {(MAX_BLOCK_DIM_X := gpu.MAX_BLOCK_DIM_X)}")
print(f"maxBlockDimY:  {(MAX_BLOCK_DIM_Y := gpu.MAX_BLOCK_DIM_Y)}")
print(f"maxBlockDimZ:  {(MAX_BLOCK_DIM_Z := gpu.MAX_BLOCK_DIM_Z)}")
print(f"maxGridDimX:  {(MAX_GRID_DIM_X := gpu.MAX_GRID_DIM_X)}")
print(f"maxGridDimY: {(MAX_GRID_DIM_Y := gpu.MAX_GRID_DIM_Y)}")
print(f"maxGridDimZ: {(MAX_GRID_DIM_Z := gpu.MAX_GRID_DIM_Z)}")
print(f"maxSharedMemoryPerBlock: {(MAX_SHARED_MEMORY_PER_BLOCK := gpu.MAX_SHARED_MEMORY_PER_BLOCK)}")
print(f"asyncEngineCount: {(ASYNC_ENGINE_COUNT := gpu.ASYNC_ENGINE_COUNT)}")
print(f"canMapHostMemory: {(CAN_MAP_HOST_MEMORY := gpu.CAN_MAP_HOST_MEMORY)}")
print(f"multiProcessorCount: {(MULTIPROCESSOR_COUNT := gpu.MULTIPROCESSOR_COUNT)}")
print(f"warpSize: {(WARP_SIZE := gpu.WARP_SIZE)}")
FREE_MEMORY, TOTAL_MEMORY = cuda.current_context().get_memory_info()
print(f"Free device memory: {FREE_MEMORY}")
print(f"Total device memory: {TOTAL_MEMORY}")
print(f"Occupied memory at cuda startup: {100*FREE_MEMORY/TOTAL_MEMORY:.2f}%")

INF = 2**32-1
VOL_EDGE = 500
RADIUS =  VOL_EDGE/2 - 0.5
VOXELS_PER_THREAD = math.ceil(VOL_EDGE/MAX_THREADS_PER_BLOCK)
THREADS_PER_BLOCK = math.ceil(VOL_EDGE/VOXELS_PER_THREAD)
A = np.zeros((VOL_EDGE, VOL_EDGE, VOL_EDGE), dtype = np.uint32)
A[VOL_EDGE//4:3*VOL_EDGE//4, VOL_EDGE//4:3*VOL_EDGE//4, VOL_EDGE//4:3*VOL_EDGE//4] = 1
print(VOXELS_PER_THREAD, THREADS_PER_BLOCK)
BASE_DEVICE_MEMORY_USE = 5000000


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


def edt_gpu_split(A, min_splits=None):
    gpu = cuda.get_current_device()
    max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
    x_dim, y_dim, z_dim = A.shape
    segments = 2
    
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
    print(A.sum())
    A.tofile("test.raw")
    return A
    
    
def edt_cpu(A):
    B = np.empty_like(A)
    single_pass_erosion_x(A, B)
    single_pass_erosion_y(B)
    single_pass_erosion_z(B)
    return B
    
    
