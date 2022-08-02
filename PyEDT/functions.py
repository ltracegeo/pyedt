import math
import logging

import numpy as np

from numba import cuda, uint16, uint32, njit, prange, get_num_threads
from numba.np.ufunc.parallel import _get_thread_id

logger = logging.getLogger("numba")
logger.setLevel(logging.ERROR)

INF = 2**32-1


############################################
### Cuda EDT Function steps
############################################
def compile_gedt_x(line_length, voxels_per_thread):
    @cuda.jit(debug=False, opt=True)
    def gedt_x(A):
        shared = cuda.shared.array(shape=(line_length, 2), dtype=uint32)
        changed = cuda.shared.array(shape=1, dtype=uint32)
        output_array = 0
        input_array = 1
        delta = uint32(1)

        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        if tx == 0:
            changed[0] = 1
        for i in range(voxels_per_thread):
            actual_tx = voxels_per_thread*tx + i
            if A[actual_tx, bx, by] == 1:
                shared[actual_tx, 0] = INF
                shared[actual_tx, 1] = INF
            else:
                shared[actual_tx, 0] = 0
                shared[actual_tx, 1] = 0
        cuda.syncthreads()    
        
        while changed[0] == 1:
            cuda.syncthreads()
            if tx == 0:
                changed[0] = 0
            cuda.syncthreads()
            output_array = 1 - output_array
            input_array = 1 - input_array
            
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                if (0 < actual_tx) and (actual_tx < (line_length-1)):
                    center_val = shared[actual_tx, input_array]
                    left_val = shared[actual_tx - 1, input_array] + delta
                    right_val = shared[actual_tx + 1, input_array] + delta
                    if left_val < center_val:
                        if left_val <= right_val:
                            shared[actual_tx, output_array] = left_val
                            if changed[0] == 0: changed[0] = 1
                        else: # left_val > right_val
                            shared[actual_tx, output_array] = right_val
                            if changed[0] == 0: changed[0] = 1
                    elif right_val < center_val:
                        shared[actual_tx, output_array] = right_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
                elif actual_tx == 0:
                    center_val = shared[actual_tx, input_array]
                    right_val = shared[actual_tx + 1, input_array] + delta
                    if right_val < center_val:
                        shared[actual_tx, output_array] = right_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
                elif actual_tx == (line_length-1):
                    center_val = shared[actual_tx, input_array]
                    left_val = shared[actual_tx - 1, input_array] + delta
                    if left_val < center_val:
                        shared[actual_tx, output_array] = left_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
            delta += 2
            cuda.syncthreads()

        for i in range(voxels_per_thread):
            actual_tx = voxels_per_thread*tx + i
            A[actual_tx, bx, by] = shared[actual_tx, input_array]
    
    return gedt_x    


def compile_gedt_y(line_length, voxels_per_thread):    
    @cuda.jit(debug=False, opt=True)
    def gedt_y(A):
        shared = cuda.shared.array(shape=(line_length, 2), dtype=uint32)
        changed = cuda.shared.array(shape=1, dtype=uint32)
        output_array = 0
        input_array = 1
        delta = uint32(1)

        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        if tx == 0:
            changed[0] = 1
        for i in range(voxels_per_thread):
            actual_tx = voxels_per_thread*tx + i
            shared[actual_tx, 0] = A[bx, actual_tx, by]
            shared[actual_tx, 1] = A[bx, actual_tx, by]
            
        cuda.syncthreads()    
        
        while changed[0] == 1:
            cuda.syncthreads()
            if tx == 0:
                changed[0] = 0
            cuda.syncthreads()
            output_array = 1 - output_array
            input_array = 1 - input_array
            
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
            
                if (0 < actual_tx) and (actual_tx < (line_length-1)):
                    center_val = shared[actual_tx, input_array]
                    left_val = shared[actual_tx - 1, input_array] + delta
                    right_val = shared[actual_tx + 1, input_array] + delta
                    if left_val < center_val:
                        if left_val <= right_val:
                            shared[actual_tx, output_array] = left_val
                            if changed[0] == 0: changed[0] = 1
                        else: # left_val > right_val
                            shared[actual_tx, output_array] = right_val
                            if changed[0] == 0: changed[0] = 1
                    elif right_val < center_val:
                        shared[actual_tx, output_array] = right_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
                elif actual_tx == 0:
                    center_val = shared[actual_tx, input_array]
                    right_val = shared[actual_tx + 1, input_array] + delta
                    if right_val < center_val:
                        shared[actual_tx, output_array] = right_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
                elif actual_tx == (line_length-1):
                    center_val = shared[actual_tx, input_array]
                    left_val = shared[actual_tx - 1, input_array] + delta
                    if left_val < center_val:
                        shared[actual_tx, output_array] = left_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
            delta += 2
            cuda.syncthreads()

        for i in range(voxels_per_thread):
            actual_tx = voxels_per_thread*tx + i
            A[bx, actual_tx, by] = shared[actual_tx, input_array]
    return gedt_y        
    

def compile_gedt_z(line_length, voxels_per_thread):   
    @cuda.jit(debug=False, opt=True)
    def gedt_z(A):
        shared = cuda.shared.array(shape=(line_length, 2), dtype=uint32)
        changed = cuda.shared.array(shape=1, dtype=uint32)
        output_array = 0
        input_array = 1
        delta = uint32(1)

        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        if tx == 0:
            changed[0] = 1
        for i in range(voxels_per_thread):
            actual_tx = voxels_per_thread*tx + i
            shared[actual_tx, 0] = A[bx, by, actual_tx]
            shared[actual_tx, 1] = A[bx, by, actual_tx]
        
        cuda.syncthreads()    
        
        while changed[0] == 1:
            cuda.syncthreads()
            if tx == 0:
                changed[0] = 0
            cuda.syncthreads()
            output_array = 1 - output_array
            input_array = 1 - input_array
            
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
            
                if (0 < actual_tx) and (actual_tx < (line_length-1)):
                    center_val = shared[actual_tx, input_array]
                    left_val = shared[actual_tx - 1, input_array] + delta
                    right_val = shared[actual_tx + 1, input_array] + delta
                    if left_val < center_val:
                        if left_val <= right_val:
                            shared[actual_tx, output_array] = left_val
                            if changed[0] == 0: changed[0] = 1
                        else: # left_val > right_val
                            shared[actual_tx, output_array] = right_val
                            if changed[0] == 0: changed[0] = 1
                    elif right_val < center_val:
                        shared[actual_tx, output_array] = right_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
                elif actual_tx == 0:
                    center_val = shared[actual_tx, input_array]
                    right_val = shared[actual_tx + 1, input_array] + delta
                    if right_val < center_val:
                        shared[actual_tx, output_array] = right_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
                elif actual_tx == (line_length-1):
                    center_val = shared[actual_tx, input_array]
                    left_val = shared[actual_tx - 1, input_array] + delta
                    if left_val < center_val:
                        shared[actual_tx, output_array] = left_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
            delta += 2
            cuda.syncthreads()

        for i in range(voxels_per_thread):
            actual_tx = voxels_per_thread*tx + i
            A[bx, by, actual_tx] = shared[actual_tx, input_array]
    return gedt_z
    
        
############################################
### CPU EDT Function steps
############################################
@njit(parallel=True)
def single_pass_erosion_x(input_array, output_array):
    w, h, d = input_array.shape
    for i in prange(w * h):
        y = i % h
        z = i // h
        
        if input_array[0, y, z] == 1:
            output_array[0, y, z] = INF
        else:
            output_array[0, y, z] = 0
            
        for x in range(1, w):
            if input_array[x, y, z] == 0:
                output_array[x, y, z] = 0
                continue
            n = output_array[x-1, y, z]
            if n == INF:
                output_array[x, y, z] = INF
            else:
                output_array[x, y, z] = n + 2*math.sqrt(n) + 1
                
        for x in range(w - 2, -1, -1):
            if input_array[x, y, z] == 1:
                n = output_array[x+1, y, z]
                if n < INF:
                    n_next = n + 2*math.sqrt(n) + 1
                    if n_next < output_array[x, y, z]:
                        output_array[x, y, z] = n_next

@njit(parallel=True)
def single_pass_erosion_y(array):
    """
    Inplace operation
    """
    w, h, d = array.shape
    
    threads_n = get_num_threads()
    work_array = np.empty((h, threads_n), dtype = np.uint32)
    
    
    for i in prange(w * d):
        x = i % w
        z = i // w
        delta = 1
        thread_index = _get_thread_id()
            
        changed = True
        output_row = work_array[:, thread_index]
        input_row = array[x, :, z]
            
        for y in range(h):
            output_row[y] = input_row[y]
        
        while changed == True:
            changed = False
            for y in range(h):
                if (y > 0) and (y < (h-1)):
                    center_val = input_row[y]
                    left_val = input_row[y - 1] + delta
                    right_val = input_row[y + 1] + delta
                    if left_val < center_val:
                        if left_val <= right_val:
                            output_row[y] = left_val
                            changed = True
                        else: # left_val > right_val
                            output_row[y] = right_val
                            changed = True
                    elif right_val < center_val:
                        output_row[y] = right_val
                        changed = True
                    else:
                        output_row[y] = center_val
                elif y == 0:
                    center_val = input_row[y]
                    right_val = input_row[y + 1] + delta
                    if right_val < center_val:
                        output_row[y] = right_val
                        changed = True
                    else:
                        output_row[y] = center_val
                elif y == (h-1):
                    center_val = input_row[y]
                    left_val = input_row[y - 1] + delta
                    if left_val < center_val:
                        output_row[y] = left_val
                        changed = True
                    else:
                        output_row[y] = center_val
            delta += 2
            output_row, input_row = input_row, output_row

        array[x, :, z] = output_row

@njit(parallel=True)
def single_pass_erosion_z(array):
    """
    Inplace operation
    """
    w, h, d = array.shape
    
    threads_n = get_num_threads()
    work_array = np.empty((d, threads_n), dtype = np.uint32)
    
    
    for i in prange(w * h):
        x = i % w
        y = i // w
        delta = 1
        thread_index = _get_thread_id()
            
        changed = True
        output_row = work_array[:, thread_index]
        input_row = array[x, y, :]
            
        for z in range(d):
            output_row[z] = input_row[z]
        
        while changed == True:
            changed = False
            for z in range(d):
                if (z > 0) and z < (d-1):
                    center_val = input_row[z]
                    left_val = input_row[z - 1] + delta
                    right_val = input_row[z + 1] + delta
                    if left_val < center_val:
                        if left_val <= right_val:
                            output_row[z] = left_val
                            changed = True
                        else: # left_val > right_val
                            output_row[z] = right_val
                            changed = True
                    elif right_val < center_val:
                        output_row[z] = right_val
                        changed = True
                    else:
                        output_row[z] = center_val
                elif z == 0:
                    center_val = input_row[z]
                    right_val = input_row[z + 1] + delta
                    if right_val < center_val:
                        output_row[z] = right_val
                        changed = True
                    else:
                        output_row[z] = center_val
                elif z == (d-1):
                    center_val = input_row[z]
                    left_val = input_row[z - 1] + delta
                    if left_val < center_val:
                        output_row[z] = left_val
                        changed = True
                    else:
                        output_row[z] = center_val
            delta += 2
            output_row, input_row = input_row, output_row

        array[x, y, :] = output_row
