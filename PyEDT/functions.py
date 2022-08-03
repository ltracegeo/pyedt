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
def single_pass_erosion_y(array, threads_n_override=0):
    """
    Inplace operation
    """
    w, h, d = array.shape
    for i in prange(w):
        for j in range(d):
            new_algorithm(array[i, :, j])
            new_algorithm(array[i, -1::-1, j])

'''
    if threads_n_override == 0:
        threads_n = max(1, get_num_threads()//2)
    else:
        threads_n = threads_n_override
    
    for thread_index in prange(threads_n):

        w, h, d = array.shape

        work_array = np.empty(h, dtype=np.uint32)
        #output_array = np.empty(h, dtype=np.uint32)
        
        for row_index in range((w*d*thread_index)//threads_n, (w*d*(thread_index + 1))//threads_n):

            array_x = row_index // w
            array_z = row_index % w
        
            x1 = 0
            y1 = array[array_x, 0, array_z]
            work_array[0] = y1
            x2 = 1
            y2 = array[array_x, 1, array_z]
            work_array[1] = y2
            if y2 < y1: 
                x3 = 0
            else: 
                x3 = math.ceil((x1*x2)/2 + (y2 - y1) / (2*(x2-x1)))
            i = 1
            for step in range(MAX_STEPS):
                if not (i < h):
                    break
                x4 = i
                y4 = array[array_x, i, array_z]
                work_array[i] = y4
                if array_x == 10 and array_z == 10: print(work_array)

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
                    if new_val < y4: work_array[i] = y1 + (i-x1)**2
                    i += 1
                    if array_x == 10 and array_z == 10: print(work_array, "anchor not changed")
                else: #change anchor
                    if array_x == 10 and array_z == 10: print(work_array, "anchor changed")
                    x1 = x2
                    y1 = y2
                    new_val = y1 + (i-x1)**2
                    if new_val < y4: work_array[i] = y1 + (i-x1)**2
                    if array_x == 10 and array_z == 10: print(work_array, "anchor changed")

                    #for i in range:
                    #    scan_next_anchor()
                    x2 = x1 + 1
                    y2 = array[0, x2, 0]
                    if y2 < y1: 
                            x3 = 0
                    else: 
                        x3 = math.ceil((x1+x2)/2 + (y2 - y1) / (2*(x2-x1)))
                            
                    for x in range(x1 + 2, i + 2):
                        if x >= h: break
                        y = array[0, x, 0]
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
                    i += 1
            else:
                raise Exception
            if array_x == 10 and array_z == 10: print(work_array)
            x1 = 0
            y1 = array[array_x, -1, array_z]
            x2 = 1
            y2 = array[array_x, -2, array_z]
            if y2 < y1: 
                x3 = 0
            else: 
                x3 = math.ceil((x1*x2)/2 + (y2 - y1) / (2*(x2-x1)))
            i = h-2
            for step in range(MAX_STEPS):
                if not (i >= 0):
                    break
                x4 = h-i
                y4 = array[array_x, i, array_z]

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
                    new_val = y1 + (x4-x1)**2
                    if new_val < work_array[i]: work_array[i] = new_val
                    i -= 1
                else: #change anchor
                    x1 = x2
                    y1 = y2
                    new_val = y1 + (x4-x1)**2
                    if new_val < work_array[i]: work_array[i] = new_val

                    x2 = x1 + 1
                    y2 = array[0, x2, 0]
                    if y2 < y1: 
                            x3 = 0
                    else: 
                        x3 = math.ceil((x1+x2)/2 + (y2 - y1) / (2*(x2-x1)))
                            
                    for x in range(x1 + 2, x4 + 2):
                        if x >= h: break
                        y = array[0, h-x-1, 0]
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
                    i -= 1
            else:
                raise Exception

            for i in range(h):
                array[array_x, i, array_z] = work_array[i]
'''
  


@njit(parallel=True)
def single_pass_erosion_z(array):
    """
    Inplace operation
    """
    w, h, d = array.shape
    for i in prange(w):
        for j in range(h):
            new_algorithm(array[i, j, :])
            new_algorithm(array[i, j, -1::-1])
            
'''
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
'''

@njit
def new_algorithm(arr):
    h = arr.shape[0]
    output = arr.copy()
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
            if new_val < arr[i]: output[i] = y1 + (i-x1)**2
            calculated_index = i
            i += 1
        else: #change anchor
            x1 = x2
            y1 = y2
            new_val = y1 + (i-x1)**2
            if new_val < arr[i]: output[i] = y1 + (i-x1)**2

            
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
    