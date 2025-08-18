## Compilation for CUDA 12.8 with GCC-14 and G++-14
sudo update-alternatives --config gcc  --> set gcc-14
sudo update-alternatives --config g++  --> set g++-14

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.8/lib64
export PATH=${PATH}:/usr/local/cuda-12.8/bin

nvcc -O3 -arch=sm_75 percentileMethods.cu -o percentileMethods
./percentileMethods

## Compilation for CUDA 11.7 with GCC-11 and G++-11
sudo update-alternatives --config gcc  --> set gcc-11
sudo update-alternatives --config g++  --> set g++-11

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11.7/lib64
export PATH=${PATH}:/usr/local/cuda-11.7/bin

nvcc -O3 -arch=sm_75 percentileMethods.cu -o percentileMethods
./percentileMethods

## Objective
Compare 2 percentile methods:
 - https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
 - https://nvidia.github.io/cccl/thrust/api/function_group__sorting_1gad39e37d88f8334cbdd3e047a53e5cfba.html

 Generate a volume of float of size 80 * 80 * 146 * 300 and pass it on both methods.
 Benchmark the memory consumed and the timing.

 ## To profile
 nsys profile --trace=cuda --cuda-memory-usage=true ./percentileMethods

 ## Results
 Stored in the NAS (https://10.37.0.103:5001/) here: /Shared/Alyson/Percentile