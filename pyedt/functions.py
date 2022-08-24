import math
import logging
import time

import numpy as np

from numba import cuda, uint16, uint32, float32, njit, prange, get_num_threads, set_num_threads
from numba.np.ufunc.parallel import _get_thread_id
from numba.typed import Dict

logger = logging.getLogger("numba")
logger.setLevel(logging.ERROR)

INF = 2**32-1
MAX_STEPS = 10**6

############################################
### Cuda EDT Function
############################################
def compile_gedt(line_length, voxels_per_thread, closed_border, axis):
    if axis not in 'xyz':
        logger.error(f"'axis' must be one of 'x', 'y', or 'z', was '{axis}'")
        return None
    @cuda.jit(debug=False, opt=True)
    def gedt(A):
        shared = cuda.shared.array(shape=(line_length, 2), dtype=uint32)
        changed = cuda.shared.array(shape=1, dtype=uint32)
        output_array = 0
        input_array = 1
        delta = uint32(1)
        
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        if axis == 'x':
            if tx == 0:
                changed[0] = 1
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                if A[actual_tx, bx, by] >= 1:
                    shared[actual_tx, 0] = INF
                    shared[actual_tx, 1] = INF
                else:
                    shared[actual_tx, 0] = 0
                    shared[actual_tx, 1] = 0
        elif axis == 'y':
            if tx == 0:
                changed[0] = 1
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                shared[actual_tx, 0] = A[bx, actual_tx, by]
                shared[actual_tx, 1] = A[bx, actual_tx, by]
        elif axis == 'z':
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
        
        if axis == 'x':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                A[actual_tx, bx, by] = shared[actual_tx, input_array]
        elif axis == 'y':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                A[bx, actual_tx, by] = shared[actual_tx, input_array]
        elif axis == 'z':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                A[bx, by, actual_tx] = shared[actual_tx, input_array]
    return gedt
        

############################################
### Cuda Anisotropic EDT Function
############################################
def compile_anisotropic_gedt(line_length, voxels_per_thread, closed_border, axis, scale):
    if axis not in 'xyz':
        logger.error(f"'axis' must be one of 'x', 'y', or 'z', was '{axis}'")
        return None
    @cuda.jit(debug=False, opt=True)
    def gedt(A):
        shared = cuda.shared.array(shape=(line_length, 2), dtype=float32)
        changed = cuda.shared.array(shape=1, dtype=float32)
        output_array = 0
        input_array = 1
        delta = float32(scale**2)
        
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        if axis == 'x':
            if tx == 0:
                changed[0] = 1
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                if A[actual_tx, bx, by] >= 1:
                    shared[actual_tx, 0] = INF
                    shared[actual_tx, 1] = INF
                else:
                    shared[actual_tx, 0] = 0
                    shared[actual_tx, 1] = 0
        elif axis == 'y':
            if tx == 0:
                changed[0] = 1
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                shared[actual_tx, 0] = A[bx, actual_tx, by]
                shared[actual_tx, 1] = A[bx, actual_tx, by]
        elif axis == 'z':
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
            delta += 2*scale**2
            cuda.syncthreads()
        
        if axis == 'x':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                A[actual_tx, bx, by] = shared[actual_tx, input_array]
        elif axis == 'y':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                A[bx, actual_tx, by] = shared[actual_tx, input_array]
        elif axis == 'z':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                A[bx, by, actual_tx] = shared[actual_tx, input_array]
    return gedt
    
    
############################################
### Cuda Multilabel EDT Function
############################################
def compile_multilabel_gedt(line_length, voxels_per_thread, closed_border, axis, scale, multilabel=False):
    if axis not in 'xyz':
        logger.error(f"'axis' must be one of 'x', 'y', or 'z', was '{axis}'")
        return None
    @cuda.jit(debug=False, opt=True)
    def gedt(A, reference):
        shared = cuda.shared.array(shape=(line_length, 2), dtype=float32)
        changed = cuda.shared.array(shape=1, dtype=float32)
        if multilabel:
            labels = cuda.shared.array(shape=(line_length,), dtype=uint32)
        output_array = 0
        input_array = 1
        delta = float32(scale**2)
        
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        if axis == 'x':
            max_tx = A.shape[0]
            if tx == 0:
                changed[0] = 1
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                if actual_tx >= max_tx: continue
                labels[actual_tx] = reference[actual_tx, bx, by]
                if A[actual_tx, bx, by] >= 1:
                    shared[actual_tx, 0] = INF
                    shared[actual_tx, 1] = INF
                else:
                    shared[actual_tx, 0] = 0
                    shared[actual_tx, 1] = 0
        elif axis == 'y':
            max_tx = A.shape[1]
            if tx == 0:
                changed[0] = 1
            for i in range(voxels_per_thread):
                if actual_tx >= max_tx: continue
                actual_tx = voxels_per_thread*tx + i
                labels[actual_tx] = reference[bx, actual_tx, by]
                shared[actual_tx, 0] = A[bx, actual_tx, by]
                shared[actual_tx, 1] = A[bx, actual_tx, by]
        elif axis == 'z':
            max_tx = A.shape[2]
            if tx == 0:
                changed[0] = 1
            for i in range(voxels_per_thread):
                if actual_tx >= max_tx: continue
                actual_tx = voxels_per_thread*tx + i
                labels[actual_tx] = reference[bx, by, actual_tx]
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
                    if multilabel:
                        if labels[actual_tx - 1] == labels[actual_tx]:
                            left_val = shared[actual_tx - 1, input_array] + delta
                        else:
                            left_val = scale
                        if labels[actual_tx + 1] == labels[actual_tx]:
                            right_val = shared[actual_tx + 1, input_array] + delta
                        else:
                            right_val = scale
                    else:
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
                    if multilabel:
                        if labels[actual_tx + 1] == labels[actual_tx]:
                            right_val = shared[actual_tx + 1, input_array] + delta
                        else:
                            right_val = scale
                    else:
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
                    if multilabel:
                        if labels[actual_tx - 1] == labels[actual_tx]:
                            left_val = shared[actual_tx - 1, input_array] + delta
                        else:
                            left_val = scale
                    else:
                        left_val = shared[actual_tx - 1, input_array] + delta
                        
                    if delta == 1 and closed_border == True:
                        left_val = min(left_val, 1)
                    if left_val < center_val:
                        shared[actual_tx, output_array] = left_val
                        if changed[0] == 0: changed[0] = 1
                    else:
                        shared[actual_tx, output_array] = center_val
            delta += 2*scale**2
            cuda.syncthreads()
        
        if axis == 'x':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                A[actual_tx, bx, by] = shared[actual_tx, input_array]
        elif axis == 'y':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                A[bx, actual_tx, by] = shared[actual_tx, input_array]
        elif axis == 'z':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                A[bx, by, actual_tx] = shared[actual_tx, input_array]
    return gedt
    
        
############################################
### CPU EDT Function steps
############################################
@njit(parallel=True, cache=True)
def single_pass_erosion(array, closed_border, axis, scale=False, sqrt_result=False):

    w, h, d = array.shape
    
    if axis == "x":
        rows = h
        columns = d
    elif axis == "y":
        rows = w
        columns = d
    elif axis == "z":
        rows = w
        columns = h
        
    reference = np.zeros(1, dtype = np.uint32)
        
    for i in prange(rows):
        for j in range(columns):
            if axis == "x":
                secondary_scan(array[:, i, j], closed_border, scale=scale)
                secondary_scan(array[-1::-1, i, j], closed_border, scale=scale)
            elif axis == "y":
                secondary_scan(array[i, :, j], closed_border, scale=scale)
                secondary_scan(array[i, -1::-1, j], closed_border, scale=scale)
            elif axis == "z":
                secondary_scan(array[i, j, :], closed_border, scale=scale)
                secondary_scan(array[i, j, -1::-1], closed_border, sqrt_result=sqrt_result, scale=scale)


@njit(parallel=True, cache=True)
def single_pass_erosion_multilabel(array, reference, closed_border, axis, scale=False, sqrt_result=False):

    w, h, d = array.shape
    
    if axis == "x":
        rows = h
        columns = d
    elif axis == "y":
        rows = w
        columns = d
    elif axis == "z":
        rows = w
        columns = h
        
    for i in prange(rows):
        for j in range(columns):
            if axis == "x":
                slice_x = (slice(None), i, j)
                reference_line = reference[slice_x]
                values = np.unique(reference_line)
            elif axis == "y":
                slice_y = (i, slice(None), j)
                reference_line = reference[slice_y]
                values = np.unique(reference_line)
            elif axis == "z":
                slice_z = (i, j, slice(None))
                reference_line = reference[slice_z]
                values = np.unique(reference_line)
            
            for val in values:
                if val == 0: continue
                if axis == "x":
                    temporary_line = where_val_val(reference_line, val, np.uint32(INF), np.uint32(0))
                    secondary_scan(temporary_line, closed_border=closed_border, scale=scale)
                    secondary_scan(temporary_line[-1::-1], closed_border=closed_border, scale=scale)
                    array[slice_x] = where_array_array(reference_line, val, temporary_line, array[slice_x])
                    
                elif axis == "y":
                    temporary_line = where_array_val(reference_line, val, array[slice_y], np.uint32(0))
                    secondary_scan(temporary_line, closed_border=closed_border, scale=scale)
                    secondary_scan(temporary_line[-1::-1], closed_border=closed_border, scale=scale)
                    array[slice_y] = where_array_array(reference_line, val, temporary_line, array[slice_y])
                
                if axis == "z":
                    temporary_line = where_array_val(reference_line, val, array[slice_z], np.uint32(0))
                    secondary_scan(temporary_line, closed_border=closed_border, scale=scale)
                    secondary_scan(temporary_line[-1::-1], closed_border=closed_border, sqrt_result=sqrt_result, scale=scale)
                    array[slice_z] = where_array_array(reference_line, val, temporary_line, array[slice_z])


@njit(cache=True)
def secondary_scan(arr, closed_border=False, sqrt_result=False, scale=False):
    h = arr.shape[0]
    output = arr.copy()
    
    if scale == False:
        scale = 1
    
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
        
    x2 = scale
    y2 = arr[1]
    
    if y2 < y1: 
        x3 = 0
    else: 
        x3 = math.ceil((x1*x2)/2 + (y2 - y1) / (2*(x2-x1)))*scale
    calculated_index = 0
    i = 1
    while (calculated_index < h) and (i < h):
        x4 = i * scale
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
            new_val = y1 + ((i*scale)-x1)**2
            if new_val < y4: output[i] = new_val
            calculated_index = i
            i += 1
        else: #change anchor
            x1 = x2
            y1 = y2
            new_val = y1 + ((i*scale)-x1)**2
            if new_val < y4: output[i] = new_val

            x2 = x1 + scale
            if round(x2/scale) < h:
                y2 = arr[round(x2/scale)]
                if y2 < y1: 
                    x3 = 0
                else: 
                    x3 = math.ceil((x1+x2)/2 + (y2 - y1) / (2*(x2-x1))) 
                    
                for i_subscan in range(round(x1/scale) + 2, i + 2):
                    if i_subscan >= h: break
                    y = arr[i_subscan]
                    if y < y2: #updates next anchor
                        x2 = i_subscan * scale 
                        y2 = y
                        if y2 < y1: 
                            x3 = 0
                        else: 
                            x3 = math.ceil((x1+x2)/2 + (y2 - y1) / (2*(x2-x1)))
                    else:
                        if y < y1: 
                            candidate_anchor_x = 0
                        else: 
                            x = i_subscan * scale
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
    if sqrt_result:
        arr[...] = np.sqrt(output).astype(np.float32).view(np.uint32)
    else:
        arr[...] = output
    
@njit(parallel=True, cache=True)
def inplace_sqrt(A):
    w, h, d = A.shape
    for i in prange(w):
        for j in range(h):
            for k in range(d):
                val = A[i, j, k]
                val = np.float32(np.sqrt(val))
                A[i, j, k] = val.view(np.uint32)
    
@njit(cache=True)
def get_label_slices(A):
    label_slices = Dict.empty(
        key_type=np.uint32,
        value_type=slice,
    )
    for i in range(A.size):
        label = A[i]
        if label not in label_slices.keys():
            label_slices = slice(i, i+1)
 
@njit(cache=True)
def where_val_val(array, com_val, true_val, false_val):
    output = np.empty_like(array)
    for i in range(array.size):
        if array[i] == com_val:
            output[i] = true_val
        else:
            output[i] = false_val
    return output
 
@njit(cache=True)
def where_array_val(array, com_val, true_array, false_val):
    output = np.empty_like(array)
    for i in range(array.size):
        if array[i] == com_val:
            output[i] = true_array[i]
        else: 
            output[i] = false_val
    return output

@njit(cache=True)
def where_array_array(array, com_val, true_array, false_array):
    output = np.empty_like(array)
    for i in range(array.size):
        if array[i] == com_val:
            output[i] = true_array[i]
        else: 
            output[i] = false_array[i]
    return output