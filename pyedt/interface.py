import math
import time
import logging

from numba import cuda, float32, uint16, uint32, njit, prange, set_num_threads
from numba.cuda.cudadrv.driver import CudaAPIError
import numpy as np

from .functions import *

logger = logging.getLogger("numba")
logger.setLevel(logging.ERROR)

BASE_DEVICE_MEMORY_USE = 5000000
BYTES_PER_PIXEL = 4
MEMORY_TOLERANCE_MARGIN = 1.1
MIN_SIZE_FOR_GPU = 1000000000


def edt_gpu(A, closed_border=False, sqrt_result=False, scale=False, multilabel=False):
    
    if scale:
        if type(scale) is not tuple:
            logger.error(f"scale must be either None or a tuple, was {scale}")
    gpu = cuda.get_current_device()
    max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
    input_2d = (A.ndim == 2)
    if input_2d:
        A = A[..., np.newaxis]

    if scale or multilabel:
        B = A.astype(np.float32)
        A_d = cuda.to_device(B)
        if scale == False:
            scale = (1.,) * 3
        elif len(scale) == 1: 
            scale = scale * 3
        if multilabel:
            reference_array = cuda.to_device(A)
            compilers = (
                lambda line_length, voxels_per_thread: compile_multilabel_gedt(line_length, voxels_per_thread, closed_border, axis='x', scale=np.float32(scale[0]), multilabel=multilabel),
                lambda line_length, voxels_per_thread: compile_multilabel_gedt(line_length, voxels_per_thread, closed_border, axis='y', scale=np.float32(scale[1]), multilabel=multilabel),
                lambda line_length, voxels_per_thread: compile_multilabel_gedt(line_length, voxels_per_thread, closed_border, axis='z', scale=np.float32(scale[2]), multilabel=multilabel),
                )
        else:
            compilers = (
                lambda line_length, voxels_per_thread: compile_anisotropic_gedt(line_length, voxels_per_thread, closed_border, axis='x', scale=np.float32(scale[0])),
                lambda line_length, voxels_per_thread: compile_anisotropic_gedt(line_length, voxels_per_thread, closed_border, axis='y', scale=np.float32(scale[1])),
                lambda line_length, voxels_per_thread: compile_anisotropic_gedt(line_length, voxels_per_thread, closed_border, axis='z', scale=np.float32(scale[2])),
                )
    else:
        A_d = cuda.to_device(A)
        compilers = (
            lambda line_length, voxels_per_thread: compile_gedt(line_length, voxels_per_thread, closed_border, axis='x'),
            lambda line_length, voxels_per_thread: compile_gedt(line_length, voxels_per_thread, closed_border, axis='y'),
            lambda line_length, voxels_per_thread: compile_gedt(line_length, voxels_per_thread, closed_border, axis='z'),
            )
    
    x_dim, y_dim, z_dim = A.shape
    
    for grid_dim_1, grid_dim_2, line_length, gedt_compiler in (
    (y_dim, z_dim, x_dim, compilers[0]),
    (x_dim, z_dim, y_dim, compilers[1]),
    (x_dim, y_dim, z_dim, compilers[2])
    ):
    
        if line_length == 1:
            continue
        voxels_per_thread = math.ceil(line_length/max_threads_per_block)
        threads_per_block = math.ceil(line_length/voxels_per_thread)
        gedt = gedt_compiler(line_length, voxels_per_thread)
        if not multilabel:
            gedt[(grid_dim_1, grid_dim_2), threads_per_block](A_d)
        else:
            gedt[(grid_dim_1, grid_dim_2), threads_per_block](A_d, reference_array)

    B = A_d.copy_to_host()
    #A.astype(np.uint16).tofile('gpu_original.raw')
    #B.astype(np.uint16).tofile('gpu_result.raw')
    #np.sqrt(B).astype(np.uint16).tofile('gpu_result_root.raw')
    #np.sqrt(B+1).astype(np.uint16).tofile('gpu_result_root_plus_one.raw')
    del A_d
    if multilabel: 
        del reference_array
    if sqrt_result:
        inplace_sqrt(B)
        B = B.view(np.float32)
    if  input_2d:
        return B[:, :, 0]
    else:
        return B


def edt_gpu_split(A, segments, closed_border=False, sqrt_result=False, scale=False, multilabel=False):
    gpu = cuda.get_current_device()
    max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
    input_2d = (A.ndim == 2)
    if input_2d:
        A = A[..., np.newaxis]
        
    if scale or multilabel:
        B = A.astype(np.float32)
        if not scale:
            scale = (1,) *3
        elif len(scale) == 1: 
            scale = scale * 3
        if multilabel:
            reference_array = cuda.to_device(A)
            compilers = (
                lambda line_length, voxels_per_thread: compile_multilabel_gedt(line_length, voxels_per_thread, closed_border, axis='x', scale=scale[0], multilabel=multilabel),
                lambda line_length, voxels_per_thread: compile_multilabel_gedt(line_length, voxels_per_thread, closed_border, axis='y', scale=scale[1], multilabel=multilabel),
                lambda line_length, voxels_per_thread: compile_multilabel_gedt(line_length, voxels_per_thread, closed_border, axis='z', scale=scale[2], multilabel=multilabel),
                )
        else:
            compilers = (
                lambda line_length, voxels_per_thread: compile_anisotropic_gedt(line_length, voxels_per_thread, closed_border, axis='x', scale=scale[0]),
                lambda line_length, voxels_per_thread: compile_anisotropic_gedt(line_length, voxels_per_thread, closed_border, axis='y', scale=scale[1]),
                lambda line_length, voxels_per_thread: compile_anisotropic_gedt(line_length, voxels_per_thread, closed_border, axis='z', scale=scale[2]),
                )
                
    else:
        B = A.copy()
        compilers = (
            lambda line_length, voxels_per_thread: compile_gedt(line_length, voxels_per_thread, closed_border, axis='x'),
            lambda line_length, voxels_per_thread: compile_gedt(line_length, voxels_per_thread, closed_border, axis='y'),
            lambda line_length, voxels_per_thread: compile_gedt(line_length, voxels_per_thread, closed_border, axis='z'),
            )
        
    
    for grid_axis_1, grid_axis_2, line_axis, gedt_compiler in (
    (1, 2, 0, compilers[0]),
    (0, 2, 1, compilers[1]),
    (0, 1, 2, compilers[2])
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
                compiled_gedts[(line_length, voxels_per_thread)] = gedt_compiler(line_length, voxels_per_thread)
            gedt = compiled_gedts[(line_length, voxels_per_thread)]
            ordered_slices = dict()
            ordered_slices[line_axis] = slices[0]
            ordered_slices[grid_axis_1] = slices[1]
            ordered_slices[grid_axis_2] = slices[2]
            A_d = cuda.to_device(np.ascontiguousarray(B[ordered_slices[0], ordered_slices[1], ordered_slices[2]]))
            if not multilabel:
                gedt[(slices[1].stop - slices[1].start, slices[2].stop - slices[2].start), threads_per_block](A_d) 
            else:
                R_d = cuda.to_device(np.ascontiguousarray(reference_array[ordered_slices[0], ordered_slices[1], ordered_slices[2]]))
                gedt[(slices[1].stop - slices[1].start, slices[2].stop - slices[2].start), threads_per_block](A_d, R_d)
            B[ordered_slices[0], ordered_slices[1], ordered_slices[2]] = A_d.copy_to_host()

    del A_d
    if multilabel: 
        del reference_array
    if sqrt_result:
        inplace_sqrt(B)
        B = B.view(np.float32)
    #B.astype(np.uint16).tofile('gpu_result.raw')
    if  input_2d:
        return B[:, :, 0]
    else:
        return B
    
    
def edt_cpu(A, closed_border=False, sqrt_result=False, limit_cpus=None, scale=False, multilabel=False):
    
    #A.astype(np.uint16).tofile("edt_cpu_A.raw")
    if limit_cpus:
        set_num_threads(limit_cpus)
    if scale:
        if len(scale) == 1: scale = scale * 3
        B = np.empty(A.shape, dtype=np.float32)
    else:
        scale = (False,) * 3
        B = np.empty(A.shape, dtype=np.uint32)
    _fill_array(A, B)
    input_2d = (B.ndim == 2)
    if input_2d:
        B = B[..., np.newaxis]
        
    if multilabel == True:
        def erosion_function(array, **kwargs):
            single_pass_erosion_multilabel(array, A, **kwargs)
    else:
        def erosion_function(array, **kwargs):
            single_pass_erosion(array, **kwargs)
        
    #B.astype(np.uint16).tofile("edt_cpu_pass_0.raw")
    #start_time_x = time.monotonic()
    erosion_function(B, closed_border=closed_border, scale=scale[0], axis="x")
    #end_time_x = time.monotonic()
    #B.astype(np.uint16).tofile("edt_cpu_pass_x.raw")
    #start_time_y = time.monotonic()
    erosion_function(B, closed_border=closed_border, scale=scale[1], axis="y")
    #end_time_y = time.monotonic()
    #B.astype(np.uint16).tofile("edt_cpu_pass_y.raw")
    #start_time_z = time.monotonic()
    if B.shape[2] > 1:
        erosion_function(B, closed_border=closed_border, sqrt_result=sqrt_result, scale=scale[2], axis="z")
    #end_time_z = time.monotonic()
    #B.astype(np.uint16).tofile("edt_cpu_pass_z.raw")
    #print(f"step times: {end_time_x - start_time_x}, {end_time_y - start_time_y}, {end_time_z - start_time_z}, total: {end_time_x - start_time_x + end_time_y - start_time_y + end_time_z - start_time_z}")
    if sqrt_result:
        B = B.view(np.float32)
    #B.astype(np.uint16).tofile('cpu_result.raw')
    if  input_2d:
        return B[:, :, 0]
    else:
        return B


@njit(parallel=True)
def _fill_array(A, B):
    if A.ndim == 3:
        w, h, d = A.shape
        for i in prange(w):
            for j in range(h):
                for k in range(d):
                    if A[i, j, k] > 0:
                        B[i, j, k] = np.uint32(INF)
                    else:
                        B[i, j, k] = 0
    else:
        w, h = A.shape
        for i in prange(w):
            for j in range(h):
                    if A[i, j] > 0:
                        B[i, j] = np.uint32(INF)
                    else:
                        B[i, j] = 0
    

def edt(A, force_method=None, minimum_segments=3, closed_border=False, sqrt_result=False, scale=False, multilabel=False):
    
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
        function = lambda a, closed_border, sqrt_result, scale, multilabel: edt_gpu_split(
            a, 
            segments, 
            closed_border=closed_border, 
            sqrt_result=sqrt_result, 
            scale=scale, 
            multilabel=multilabel
        )

    return function(
        A, 
        closed_border=closed_border, 
        sqrt_result=sqrt_result, 
        scale=scale, 
        multilabel=multilabel
    )


def run_benchmark(size_override=None, 
                  plot=False, 
                  test_sqrt=False, 
                  other_modules_benchmark=True,
                  gpu_split_segments = (2,3,4)
                  ):
    try:
        from scipy import ndimage
    except ModuleNotFoundError as e:
        logging.info("failed to load scipy", e)
        ndimage_loaded = False
    else:
        ndimage_loaded = other_modules_benchmark
       
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as e:
        logging.info("failed to load SimpleITK", e)
        sitk_loaded = False
    else:
        sitk_loaded = other_modules_benchmark

    try:
        import edt as edt_ws
    except ModuleNotFoundError as e:
        logging.info("failed to load edt", e)
        edt_loaded = False
    else:
        edt_loaded = other_modules_benchmark

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
            if test_sqrt:
                try:
                    start_time = time.monotonic()
                    _ = edt(A, force_method=method, sqrt_result=True)
                    end_time = time.monotonic()
                except Exception as e:
                    logging.info(method, size, e)
                    results[f"{method}_sqrt_{size}"] = "fail"
                else:
                    results[f"{method}_sqrt_{size}"] = end_time - start_time
        for segments in gpu_split_segments:
            try:
                start_time = time.monotonic()
                _ = edt(A, force_method="gpu-split", minimum_segments = segments)
                end_time = time.monotonic()
            except Exception as e:
                logging.info("gpu-split", size, e)
                results[f"gpu-split_{segments}_{size}"] = "fail"
            else:
                results[f"gpu-split_{segments}_{size}"] = end_time - start_time
            if test_sqrt:
                try:
                    start_time = time.monotonic()
                    _ = edt(A, force_method="gpu-split", minimum_segments=segments, sqrt_result=True)
                    end_time = time.monotonic()
                except Exception as e:
                    logging.info("gpu-split", size, e)
                    results[f"gpu-split_{segments}_sqrt_{size}"] = "fail"
                else:
                    results[f"gpu-split_{segments}_sqrt_{size}"] = end_time - start_time

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
        cpu_names = [i for i in results.keys() if "cpu" in i and "sqrt" not in i] 
        cpu_sqrt_names = [i for i in results.keys() if "cpu" in i and "sqrt" in i]
        gpu_names = [i for i in results.keys() if "gpu" in i and "split" not in i and "sqrt" not in i]
        gpu_sqrt_names = [i for i in results.keys() if "gpu" in i and "split" not in i and "sqrt" in i]
        gpu_split_2_names = [i for i in results.keys() if "gpu-split_2" in i and "sqrt" not in i]  
        gpu_sqrt_split_2_names = [i for i in results.keys() if "gpu-split_2" in i and "sqrt" in i]        
        gpu_split_3_names = [i for i in results.keys() if "gpu-split_3" in i and "sqrt" not in i]
        gpu_sqrt_split_3_names = [i for i in results.keys() if "gpu-split_3" in i and "sqrt" in i]
        gpu_split_4_names = [i for i in results.keys() if "gpu-split_4" in i and "sqrt" not in i]
        gpu_sqrt_split_4_names = [i for i in results.keys() if "gpu-split_4" in i and "sqrt" in i]
        ndimage_names = [i for i in results.keys() if "ndimage" in i]
        sitk_names = [i for i in results.keys() if "sitk" in i]
        edt_names = [i for i in results.keys() if "edt" in i]
        values = [cpu_names, 
                  cpu_sqrt_names,
                  gpu_names,
                  gpu_sqrt_names,
                  gpu_split_2_names,
                  gpu_sqrt_split_2_names,
                  gpu_split_3_names,
                  gpu_sqrt_split_3_names,
                  gpu_split_4_names,
                  gpu_sqrt_split_4_names,
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
            
    
def _auto_decide_method(A, multilabel=False):
    if not cuda.is_available():
        return "cpu"
        
    free_memory, total_memory = cuda.current_context().get_memory_info()
    expected_memory_use = (A.size*BYTES_PER_PIXEL + BASE_DEVICE_MEMORY_USE) * MEMORY_TOLERANCE_MARGIN
    
    gpu = cuda.get_current_device()
    MAX_GRID_DIM_X = gpu.MAX_GRID_DIM_X
    MAX_GRID_DIM_Y = gpu.MAX_GRID_DIM_Y
    MAX_GRID_DIM_Z = gpu.MAX_GRID_DIM_Z
    MAX_SHARED_MEMORY_PER_BLOCK = gpu.MAX_SHARED_MEMORY_PER_BLOCK
    
    max_axis = np.max(A.shape)
    max_grid = np.max((MAX_GRID_DIM_X, MAX_GRID_DIM_Y, MAX_GRID_DIM_Z))
    
    if ((MAX_SHARED_MEMORY_PER_BLOCK < (3 * (max_axis+1) * BYTES_PER_PIXEL * MEMORY_TOLERANCE_MARGIN)) or
       (max_grid < max_axis)):
       return "cpu"
    
    if free_memory > expected_memory_use:
        return "gpu"
        
    return "gpu-split"
auto_decide_method = _auto_decide_method # fix for pytest
