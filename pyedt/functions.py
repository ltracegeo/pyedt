import logging
import math
import numpy as np

from numba import cuda, uint32, float32, njit, prange
from numba.typed import Dict


logger = logging.getLogger("numba")
logger.setLevel(logging.ERROR)

INF = 2**31-1
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
                if (0 <= actual_tx) and (actual_tx < line_length):
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
                if (0 <= actual_tx) and (actual_tx < line_length):
                    shared[actual_tx, 0] = A[bx, actual_tx, by]
                    shared[actual_tx, 1] = A[bx, actual_tx, by]
        elif axis == 'z':
            if tx == 0:
                changed[0] = 1
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                if (0 <= actual_tx) and (actual_tx < line_length):
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
                if (0 <= actual_tx) and (actual_tx < line_length):
                    A[actual_tx, bx, by] = shared[actual_tx, input_array]
        elif axis == 'y':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                if (0 <= actual_tx) and (actual_tx < line_length):
                    A[bx, actual_tx, by] = shared[actual_tx, input_array]
        elif axis == 'z':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                if (0 <= actual_tx) and (actual_tx < line_length):
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
                if (0 <= actual_tx) and (actual_tx < line_length):
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
                if (0 <= actual_tx) and (actual_tx < line_length):
                    shared[actual_tx, 0] = A[bx, actual_tx, by]
                    shared[actual_tx, 1] = A[bx, actual_tx, by]
        elif axis == 'z':
            if tx == 0:
                changed[0] = 1
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                if (0 <= actual_tx) and (actual_tx < line_length):
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
                if (0 <= actual_tx) and (actual_tx < line_length):
                    A[actual_tx, bx, by] = shared[actual_tx, input_array]
        elif axis == 'y':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                if (0 <= actual_tx) and (actual_tx < line_length):
                    A[bx, actual_tx, by] = shared[actual_tx, input_array]
        elif axis == 'z':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                if (0 <= actual_tx) and (actual_tx < line_length):
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
                if (0 <= actual_tx) and (actual_tx < line_length):
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
                actual_tx = voxels_per_thread*tx + i
                if actual_tx >= max_tx: continue
                if (0 <= actual_tx) and (actual_tx < line_length):
                    labels[actual_tx] = reference[bx, actual_tx, by]
                    shared[actual_tx, 0] = A[bx, actual_tx, by]
                    shared[actual_tx, 1] = A[bx, actual_tx, by]
        elif axis == 'z':
            max_tx = A.shape[2]
            if tx == 0:
                changed[0] = 1
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread * tx + i
                if actual_tx >= max_tx: continue
                if (0 <= actual_tx) and (actual_tx < line_length):
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
                if (0 <= actual_tx) and (actual_tx < line_length):
                    A[actual_tx, bx, by] = shared[actual_tx, input_array]
        elif axis == 'y':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                if (0 <= actual_tx) and (actual_tx < line_length):
                    A[bx, actual_tx, by] = shared[actual_tx, input_array]
        elif axis == 'z':
            for i in range(voxels_per_thread):
                actual_tx = voxels_per_thread*tx + i
                if (0 <= actual_tx) and (actual_tx < line_length):
                    A[bx, by, actual_tx] = shared[actual_tx, input_array]
    return gedt
    
        
############################################
### CPU EDT Function steps
############################################
@njit(parallel=True, cache=True)
def single_pass_erosion(array, closed_border, axis, scale=False):

    w, h, d = array.shape
    do_print = False
    print_x = 107
    print_y = 56
    print_z = 131
    
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
                pr = (i == print_y) and (j == print_z) and do_print
                if pr: print(array[:, i, j])
                secondary_scan(array[:, i, j], closed_border, scale=scale, pr=pr)
                if pr: print(array[:, i, j])
                secondary_scan(array[-1::-1, i, j], closed_border, scale=scale, pr=pr)
                if pr: print(array[:, i, j])
            elif axis == "y":
                pr = (i == print_x) and (j == print_z) and do_print
                if pr: print(array[i, :, j])
                secondary_scan(array[i, :, j], closed_border, scale=scale, pr=pr)
                if pr: print(array[i, :, j])
                secondary_scan(array[i, -1::-1, j], closed_border, scale=scale, pr=pr)
                if pr: print(array[i, :, j])
            elif axis == "z":
                pr = (i == print_x) and (j == print_y) and do_print
                if pr: print(array[i, j, :])
                secondary_scan(array[i, j, :], closed_border, scale=scale, pr=pr)
                if pr: print(array[i, j, :])
                secondary_scan(array[i, j, -1::-1], closed_border, scale=scale, pr=pr)
                if pr: print(array[i, j, :])

@njit(parallel=False, cache=True)
def single_pass_erosion_serial(array, closed_border, axis, scale=False):

    w, h, d = array.shape
    do_print = False
    print_x = 107
    print_y = 56
    print_z = 131
    
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
                pr = (i == print_y) and (j == print_z) and do_print
                if pr: print(array[:, i, j])
                secondary_scan(array[:, i, j], closed_border, scale=scale, pr=pr)
                if pr: print(array[:, i, j])
                secondary_scan(array[-1::-1, i, j], closed_border, scale=scale, pr=pr)
                if pr: print(array[:, i, j])
            elif axis == "y":
                pr = (i == print_x) and (j == print_z) and do_print
                if pr: print(array[i, :, j])
                secondary_scan(array[i, :, j], closed_border, scale=scale, pr=pr)
                if pr: print(array[i, :, j])
                secondary_scan(array[i, -1::-1, j], closed_border, scale=scale, pr=pr)
                if pr: print(array[i, :, j])
            elif axis == "z":
                pr = (i == print_x) and (j == print_y) and do_print
                if pr: print(array[i, j, :])
                secondary_scan(array[i, j, :], closed_border, scale=scale, pr=pr)
                if pr: print(array[i, j, :])
                secondary_scan(array[i, j, -1::-1], closed_border, scale=scale, pr=pr)
                if pr: print(array[i, j, :])


@njit(parallel=True, cache=True)
def single_pass_erosion_multilabel(array, reference, closed_border, axis, scale=False):

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
                    secondary_scan(temporary_line[-1::-1], closed_border=closed_border, scale=scale)
                    array[slice_z] = where_array_array(reference_line, val, temporary_line, array[slice_z])

@njit(cache=True)
def get_x3(x1, y1, x2, y2):
    if y2 > y1:
        return (x1 + x2)/2 + ((y1 + y2)/2) / ((x2-x1)/(y2-y1))
    else:
        return (x1 + x2)/2

@njit(cache=True)
def secondary_scan(arr, closed_border=False, scale=False, pr=False):
    h = arr.shape[0]
    output = arr.copy()

    if scale == False:
        scale = 1
    
    '''
    (x1, y1) --> Current anchor
    (x2, y2) --> Next anchor
    (x3, __) --> Anchor exchange point
    (x4, y4) --> Evaluated point
    '''
    
    x1 = -scale
    if closed_border:
        y1 = 0
    else:
        y1 = np.sqrt(INF)
        
    x2 = 0
    y2 = np.sqrt(arr[0])
    
    output[0] = min(arr[0], (y1 + scale)**2)
    if arr[0] < (y1 + scale)**2:
        output[0] = arr[0]
    else:
        output[0] = (y1 + scale)**2
    if pr: print("i0", output[0], arr[0], y1, scale)

    i = 1
    while (i < h):
        x3 = get_x3(x1, y1, x2, y2)
        
        x4 = i * scale
        y4 = np.sqrt(arr[i])
        
        #if pr: print(f"{i} start: ({x1}, {y1})  ({x2}, {y2})  ({x3}, __)  ({x4}, {y4}) (d: {output[i]})")
        if pr: print(i,x1,y1,x2,y2,x3,x4,y4,output[i],"\n")
        
        if x4 >= x3:
            x1 = x2
            y1 = y2
            i2 = int(np.round(x1/scale)+1)
            x2 = i2 * scale
            if i2 < h:
                y2 = np.sqrt(arr[i2])
            else:
                y2 = 0
            x3 = get_x3(x1, y1, x2, y2)
            
            # scan candidate anchors
            for anchor_i in range(i2, i+1):
                if anchor_i >= h-1: continue
                candidate_x = anchor_i * scale
                candidate_y = np.sqrt(arr[anchor_i])
                candidate_x3 = get_x3(x1, y1, candidate_x, candidate_y)
                if pr: print("i: ", anchor_i, candidate_x, candidate_y, candidate_x3,"\n")
                if candidate_x3 <= x4:
                    x1 = candidate_x
                    y1 = candidate_y
                    x2 = (anchor_i+1) * scale
                    y2 = np.sqrt(arr[anchor_i+1])
                    x3 = get_x3(x1, y1, x2, y2)
                elif candidate_x3 <= x3:
                    x2 = candidate_x
                    y2 = candidate_y
                    x3 = candidate_x3
                    
        else:
            candidate_x = x4
            candidate_y = y4
            candidate_x3 = get_x3(x1, y1, candidate_x, candidate_y)
            if candidate_x3 <= x3:
                x2 = candidate_x
                y2 = candidate_y
                x3 = candidate_x3
        
        if y1 > 30000 or (x4 - x1) > 30000:
            new_distance = INF
        else:
            new_distance = np.uint32(np.round(y1**2 + (x4 - x1)**2))
        
        if new_distance < output[i]:
            output[i] = new_distance
        
        #if pr: print(f"{i} end:({x1}, {y1})  ({x2}, {y2})  ({x3}, __)  ({x4}, {y4}) (d: {output[i]} / {new_distance})\n")
        if pr: print(i,x1,y1,x2,y2,x3,x4,y4,output[i],new_distance,"\n")
        #calculated_index = i
        i += 1
        
    if pr:
        print("#######\nWrong Values:")
        wrong_indexes_i, wrong_indexes_j, expected_values, found_values = check_secondary_scan(arr, output, closed_border)
        for j in range(len(wrong_indexes_i)):
            print(wrong_indexes_i[j], wrong_indexes_j[j], expected_values[j], found_values[j])
        print("End wrong values")
        
    arr[...] = output
    
    
@njit(parallel=True, cache=True)
def inplace_sqrt_uint32(A):
    w, h, d = A.shape
    for i in prange(w):
        for j in range(h):
            for k in range(d):
                val = A[i, j, k]
                val = np.float32(np.sqrt(val))
                A[i, j, k] = val.view(np.uint32)

@njit(parallel=True, cache=True)
def inplace_sqrt_float32(A):
    w, h, d = A.shape
    for i in prange(w):
        for j in range(h):
            for k in range(d):
                A[i, j, k] = np.sqrt(A[i, j, k])

@njit(cache=True)
def inplace_sqrt_float32_serial(A):
    w, h, d = A.shape
    for i in range(w):
        for j in range(h):
            for k in range(d):
                A[i, j, k] = np.sqrt(A[i, j, k])

def inplace_sqrt(A):
    if A.dtype == np.uint32:
        inplace_sqrt_uint32(A)
    elif A.dtype == np.float32:
        inplace_sqrt_float32(A)
    else:
        logger.error(f"Array must be of type uint32 or float32, was {A.dtype}")

@njit(parallel=True, cache=True)
def as_contiguous(A):
    w, h, d = A.shape
    B = np.empty_like(A)
    for i in prange(w):
        for j in range(h):
            for k in range(d):
                B[i, j, k] = A[i, j, k]
    return B
    
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
    
@njit(cache=True)
def check_secondary_scan(input_array, output_array, closed_border=False):
    wrong_indexes_i = []
    wrong_indexes_j = []
    expected_values = []
    found_values = []
    
    for i in range(len(input_array)):
        lowest_distance_i = i
        lowest_distance_j = INF
        if closed_border:
            lowest_distance = (i+1)**2
        else:
            lowest_distance = INF
        for j in range(i+1):
            distance = input_array[j] + (i-j)**2
            if distance < lowest_distance:
                lowest_distance_i = i
                lowest_distance_j = j
                lowest_distance = distance
        if lowest_distance != output_array[i]:
            wrong_indexes_i.append(lowest_distance_i)
            wrong_indexes_j.append(lowest_distance_j)
            expected_values.append(lowest_distance)
            found_values.append(output_array[i])
            
    return wrong_indexes_i, wrong_indexes_j, expected_values, found_values
                
            