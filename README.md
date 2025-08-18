## Compilation
nvcc -arch=sm_75 percentileMethods.cu -o percentileMethods
./percentileMethods

## Objective
Compare 2 percentile methods:
 - https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
 - https://nvidia.github.io/cccl/thrust/api/function_group__sorting_1gad39e37d88f8334cbdd3e047a53e5cfba.html

 Generate a volume of float of size 80*80*146*300 and pass it on both methods.
 Benchmark the memory consumed and the timing.