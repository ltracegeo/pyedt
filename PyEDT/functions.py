import math
import logging
import time

import numpy as np

from numba import cuda, uint16, uint32, njit, prange, get_num_threads
from numba.np.ufunc.parallel import _get_thread_id

logger = logging.getLogger("numba")
logger.setLevel(logging.ERROR)

INF = 2**32-1
MAX_STEPS = 10**6

############################################
### Cuda EDT Function steps
############################################
def compile_gedt_x(line_length, voxels_per_thread, closed_border):
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
                    if delta == 1 and closed_border == True:
                        right_val = min(right_val, 1)
                    if right_val < center_val:
                        shared[actual_tx, output_array] = right_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
                elif actual_tx == (line_length-1):
                    center_val = shared[actual_tx, input_array]
                    left_val = shared[actual_tx - 1, input_array] + delta
                    if delta == 1 and closed_border == True:
                        left_val = min(left_val, 1)
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


def compile_gedt_y(line_length, voxels_per_thread, closed_border):    
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
                    if delta == 1 and closed_border == True:
                        right_val = min(right_val, 1)
                    if right_val < center_val:
                        shared[actual_tx, output_array] = right_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
                elif actual_tx == (line_length-1):
                    center_val = shared[actual_tx, input_array]
                    left_val = shared[actual_tx - 1, input_array] + delta
                    if delta == 1 and closed_border == True:
                        left_val = min(left_val, 1)
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
    

def compile_gedt_z(line_length, voxels_per_thread, closed_border):   
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
                    if delta == 1 and closed_border == True:
                        right_val = min(right_val, 1)
                    if right_val < center_val:
                        shared[actual_tx, output_array] = right_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
                elif actual_tx == (line_length-1):
                    center_val = shared[actual_tx, input_array]
                    left_val = shared[actual_tx - 1, input_array] + delta
                    if delta == 1 and closed_border == True:
                        left_val = min(left_val, 1)
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
#@njit(parallel=True)
def single_pass_erosion_x(array, closed_border):
    """
    Inplace operation
    """
    w, h, d = array.shape
    for i in prange(h):
        for j in range(d):
            secondary_scan(array[:, i, j], closed_border)
            secondary_scan(array[-1::-1, i, j], closed_border)
            

#@njit(parallel=True)
def single_pass_erosion_y(array, closed_border):
    """
    Inplace operation
    """
    w, h, d = array.shape
    for i in prange(w):
        for j in range(d):
            secondary_scan(array[i, :, j], closed_border)
            secondary_scan(array[i, -1::-1, j], closed_border)


#@njit(parallel=True)
def single_pass_erosion_z(array, closed_border):
    """
    Inplace operation
    """
    w, h, d = array.shape
    for i in prange(w):
        for j in range(h):
            secondary_scan(array[i, j, :], closed_border)
            secondary_scan(array[i, j, -1::-1], closed_border)


#@njit
def secondary_scan(arr, closed_border=False):
    h = arr.shape[0]
    output = arr.copy()
    
    if closed_border:
        if arr[0] > 1:
            output[0] = 1
            x1 = -1
            y1 = 0
        else:
            x1 = 0
            y1 = arr[0]
    else:
        x1 = 0
        y1 = arr[0]
    x2 = 1
    y2 = arr[1]
    if y2 < y1: 
        x3 = 0
    else: 
        x3 = math.ceil((x1*x2)/2 + (y2 - y1) / (2*(x2-x1)))
    calculated_index = 0
    i = 1
    while (calculated_index < h) and (i < h):
        x4 = i
        y4 = arr[i]
        if x4 <= x3: # next anchor is not triggered
            #check if next anchor can be changed for current point
            if y4 < y2: #updates next anchor
                x2 = x4 
                y2 = y4
                if y2 < y1: 
                    x3 = 0
                else: 
                    x3 = math.ceil((x1+x2)/2 + (y2 - y1) / (2*(x2-x1)))
            else:
                if y4 < y1: 
                    candidate_anchor_x = 0
                else: 
                    candidate_anchor_x = math.ceil((x1+x4)/2 + (y4 - y1) / (2*(x4-x1)))
                if candidate_anchor_x < x3: #updates next anchor
                    x2 = x4
                    y2 = y4
                    x3 = candidate_anchor_x
                elif candidate_anchor_x == x3:
                    current_anchor_at_x = y2 + (x3-x2)**2
                    candidate_anchor_at_x = y4 + (x3-x4)**2
                    if candidate_anchor_at_x < current_anchor_at_x: #updates next anchor
                        x2 = x4
                        y2 = y4
                        x3 = candidate_anchor_x

        # must check (x4 <= x3) again, since last update may have changed x3
        if x4 < x3: # keep anchor
            new_val = y1 + (i-x1)**2
            if new_val < y4: output[i] = y1 + (i-x1)**2
            calculated_index = i
            i += 1
        else: #change anchor
            x1 = x2
            y1 = y2
            new_val = y1 + (i-x1)**2
            if new_val < y4: output[i] = y1 + (i-x1)**2

            x2 = x1 + 1
            if x2 < h:
                y2 = arr[x2]
                if y2 < y1: 
                        x3 = 0
                else: 
                    x3 = math.ceil((x1+x2)/2 + (y2 - y1) / (2*(x2-x1))) 
                    
                for x in range(x1 + 2, i + 2):
                    if x >= h: break
                    y = arr[x]
                    if y < y2: #updates next anchor
                        x2 = x 
                        y2 = y
                        if y2 < y1: 
                            x3 = 0
                        else: 
                            x3 = math.ceil((x1+x2)/2 + (y2 - y1) / (2*(x2-x1)))
                    else:
                        if y < y1: 
                            candidate_anchor_x = 0
                        else: 
                            candidate_anchor_x = math.ceil((x+x1)/2 + (y - y1) / (2*(x-x1)))
                        if candidate_anchor_x < x3: #updates next anchor
                            x2 = x
                            y2 = y
                            x3 = candidate_anchor_x
                        elif candidate_anchor_x == x3:
                            current_anchor_at_x = y2 + (x3-x2)**2
                            candidate_anchor_at_x = y + (x3-x)**2
                            if candidate_anchor_at_x < current_anchor_at_x: #updates next anchor
                                x2 = x
                                y2 = y
                                x3 = candidate_anchor_x
            
            calculated_index = i
            i += 1
            
    arr[...] = output
    