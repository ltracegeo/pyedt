import math
import time
import logging

from numba import cuda, float32, uint16, uint32, njit, prange
from numba.cuda.cudadrv.driver import CudaAPIError
import numpy as np

from .functions import *


BASE_DEVICE_MEMORY_USE = 5000000
BYTES_PER_PIXEL = 4
MEMORY_TOLERANCE_MARGIN = 1.1
MIN_SIZE_FOR_GPU = 1000000000


def edt_gpu(A, closed_border=False):
    gpu = cuda.get_current_device()
    max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
    input_2d = (A.ndim == 2)
    if input_2d:
        A = A[..., np.newaxis]
    A_d = cuda.to_device(A)
    x_dim, y_dim, z_dim = A.shape
    for grid_dim_1, grid_dim_2, line_length, gedt_compiler in (
    (y_dim, z_dim, x_dim, compile_gedt_x),
    (x_dim, z_dim, y_dim, compile_gedt_y),
    (x_dim, y_dim, z_dim, compile_gedt_z)
    ):
        if line_length == 1:
            continue
        voxels_per_thread = math.ceil(line_length/max_threads_per_block)
        threads_per_block = math.ceil(line_length/voxels_per_thread)
        gedt = gedt_compiler(line_length, voxels_per_thread, closed_border)
        gedt[(grid_dim_1, grid_dim_2), threads_per_block](A_d)

    B = A_d.copy_to_host()
    del A_d
    if  input_2d:
        return B[:, :, 0]
    else:
        return B


def edt_gpu_split(A, segments, closed_border=False):
    gpu = cuda.get_current_device()
    max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
    input_2d = (A.ndim == 2)
    if input_2d:
        A = A[..., np.newaxis]
    B = A.copy()
    for grid_axis_1, grid_axis_2, line_axis, gedt_compiler in (
    (1, 2, 0, compile_gedt_x),
    (0, 2, 1, compile_gedt_y),
    (0, 1, 2, compile_gedt_z)
    ):  
        segments_1 = segments
        segments_2 = segments
        grid_dim_1 = B.shape[grid_axis_1]
        if grid_dim_1 == 1:
            segments_1 = 1
        grid_dim_2 = B.shape[grid_axis_2]
        if grid_dim_2 == 1:
            segments_2 = 1
        line_length = B.shape[line_axis]
        if line_length == 1:
            continue
        
        compiled_gedts = dict()
        voxels_per_thread = math.ceil(line_length/max_threads_per_block)
        threads_per_block = math.ceil(line_length/voxels_per_thread)
    
        slices_tuple = [(slice(None),
                         slice((i)*grid_dim_1//(segments_1), (i+1)*grid_dim_1//(segments_1)),
                         slice((j)*grid_dim_2//(segments_2), (j+1)*grid_dim_2//(segments_2)))
                         for i in range(segments_1) for j in range(segments_2)]

        for slices in slices_tuple:
            if (line_length, voxels_per_thread) not in compiled_gedts.keys():
                compiled_gedts[(line_length, voxels_per_thread)] = gedt_compiler(line_length, voxels_per_thread, closed_border)
            gedt = compiled_gedts[(line_length, voxels_per_thread)]
            ordered_slices = dict()
            ordered_slices[line_axis] = slices[0]
            ordered_slices[grid_axis_1] = slices[1]
            ordered_slices[grid_axis_2] = slices[2]
            A_d = cuda.to_device(np.ascontiguousarray(B[ordered_slices[0], ordered_slices[1], ordered_slices[2]]))
            gedt[(slices[1].stop - slices[1].start, slices[2].stop - slices[2].start), threads_per_block](A_d)    
            B[ordered_slices[0], ordered_slices[1], ordered_slices[2]] = A_d.copy_to_host()

    del A_d
    if  input_2d:
        return B[:, :, 0]
    else:
        return B
    
    
def edt_cpu(A, closed_border=False):
    
    B = np.where(A > 0, INF, 0)
    input_2d = (B.ndim == 2)
    if input_2d:
        B = B[..., np.newaxis]
    #B.astype(np.uint16).tofile("edt_cpu_pass_0.raw")
    #start_time_x = time.monotonic()
    single_pass_erosion_x(B, closed_border)
    #end_time_x = time.monotonic()
    #B.astype(np.uint16).tofile("edt_cpu_pass_x.raw")
    #start_time_y = time.monotonic()
    single_pass_erosion_y(B, closed_border)
    #end_time_y = time.monotonic()
    #B.astype(np.uint16).tofile("edt_cpu_pass_y.raw")
    #start_time_z = time.monotonic()
    if B.shape[2] > 1:
        single_pass_erosion_z(B, closed_border)
    #end_time_z = time.monotonic()
    #B.astype(np.uint16).tofile("edt_cpu_pass_z.raw")
    #print(f"step times: {end_time_x - start_time_x}, {end_time_y - start_time_y}, {end_time_z - start_time_z}, total: {end_time_x - start_time_x + end_time_y - start_time_y + end_time_z - start_time_z}")
    if  input_2d:
        return B[:, :, 0]
    else:
        return B
    

def edt(A, force_method=None, minimum_segments=3, closed_border=False):
    
    if A.dtype != np.uint32:
        A = A.astype(np.uint32)
    if force_method == None:
        method = _auto_decide_method(A)
    elif force_method in ('cpu', 'gpu', 'gpu-split'):
        method = force_method
    else:
        raise ValueError(f"force_method must be one of 'cpu', 'gpu' or 'gpu-split', was {force_method}")
        
    if method == "cpu":
        logging.info("using cpu")
        function = edt_cpu
    elif method == "gpu":
        logging.info("using gpu")
        function = edt_gpu
    elif method == "gpu-split":
        free_memory, total_memory = cuda.current_context().get_memory_info()
        expected_memory_use = (A.size*BYTES_PER_PIXEL + BASE_DEVICE_MEMORY_USE) * MEMORY_TOLERANCE_MARGIN
        segments = math.ceil(math.sqrt(expected_memory_use/free_memory))
        if minimum_segments:
            segments = max(segments, minimum_segments)
        logging.info(f"using gpu {segments} segments")
        function = lambda x, y: edt_gpu_split(x, segments, y)

    return function(A, closed_border)


def run_benchmark(size_override=None, plot=False):
    try:
        from scipy import ndimage
    except ModuleNotFoundError as e:
        logging.info("failed to load scipy", e)
        ndimage_loaded = False
    else:
        ndimage_loaded = True
       
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as e:
        logging.info("failed to load SimpleITK", e)
        sitk_loaded = False
    else:
        sitk_loaded = True

    try:
        import edt as edt_ws
    except ModuleNotFoundError as e:
        logging.info("failed to load edt", e)
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
                logging.info(method, size, e)
                results[f"{method}_{size}"] = "fail"
            else:
                results[f"{method}_{size}"] = end_time - start_time
        for segments in (2, 3, 4):
            try:
                start_time = time.monotonic()
                _ = edt(A, force_method="gpu-split", minimum_segments = segments)
                end_time = time.monotonic()
            except Exception as e:
                logging.info("gpu-split", size, e)
                results[f"gpu-split_{segments}_{size}"] = "fail"
            else:
                results[f"gpu-split_{segments}_{size}"] = end_time - start_time

        if ndimage_loaded:
            try:
                start_time = time.monotonic()
                _ = ndimage.distance_transform_edt(A)
                end_time = time.monotonic()
            except Exception as e:
                logging.info("ndimage", size, e)
                results[f"ndimage_{size}"] = "fail"
            else:
                results[f"ndimage_{size}"] = end_time - start_time
        
        if sitk_loaded:
            try:
                start_time = time.monotonic()
                _ = sitk.SignedMaurerDistanceMap(sitk.GetImageFromArray(A))
                end_time = time.monotonic()
            except Exception as e:
                logging.info("sitk", size, e)
                results[f"sitk_{size}"] = "fail"
            else:
                results[f"sitk_{size}"] = end_time - start_time
            
        if edt_loaded:
            try:
                start_time = time.monotonic()
                _ = edt_ws.edt(A)
                end_time = time.monotonic()
            except Exception as e:
                logging.info("edt", size, e)
                results[f"edt_{size}"] = "fail"
            else:
                results[f"edt_{size}"] = end_time - start_time
    
    if plot:
        import matplotlib.pyplot as plt
        ax = plt.subplot(111)
        ax.set_yscale('log')
        cpu_names = [i for i in results.keys() if "cpu" in i] 
        gpu_names = [i for i in results.keys() if "gpu" in i and "split" not in i]
        gpu_split_2_names = [i for i in results.keys() if "gpu-split_2" in i]   
        gpu_split_3_names = [i for i in results.keys() if "gpu-split_3" in i]
        gpu_split_4_names = [i for i in results.keys() if "gpu-split_4" in i]
        ndimage_names = [i for i in results.keys() if "ndimage" in i]
        sitk_names = [i for i in results.keys() if "sitk" in i]
        edt_names = [i for i in results.keys() if "edt" in i]
        values = [cpu_names, 
                  gpu_names, 
                  gpu_split_2_names, 
                  gpu_split_3_names, 
                  gpu_split_4_names, 
                  ndimage_names, 
                  sitk_names, 
                  edt_names]
        values = [i for i in values if len(i) > 0]
        for val in values:
            ax.plot(list(int(i.split("_")[-1]) for i in val if results[i] != "fail"),
                    list(results[i] for i in val if results[i] != "fail"), 
                    "o--",
                    label=val[0][:val[0].rfind("_")])
        plt.xlabel("Volume edge size")
        plt.ylabel("Run time [s]")
        plt.legend()
        plt.show()
    
    return results
            
    
def _auto_decide_method(A):
    if not cuda.is_available():
        return "cpu"
        
    free_memory, total_memory = cuda.current_context().get_memory_info()
    expected_memory_use = (A.size*BYTES_PER_PIXEL + BASE_DEVICE_MEMORY_USE) * MEMORY_TOLERANCE_MARGIN
    
    if free_memory > expected_memory_use:
        return "gpu"
        
    return "gpu-split"
