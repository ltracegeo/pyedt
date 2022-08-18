# pyedt
Python Euclidian distance transform with numba and cuda

![Benchmarks](doc/benchmarks.png)

Legend
 * cpu - PyEDT running on cpu only
 * gpu - PyEDT running with CUDA, sending the whole image to the GPU
 * gpu_split_n - PyEDT running with CUDA, the image is split in n^2 prisms and processed one at a time in the GPU
