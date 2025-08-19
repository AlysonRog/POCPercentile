## Set compilation and runtime environment for CUDA 12.8 with GCC-14 and G++-14
sudo update-alternatives --config gcc  --> set gcc-14
sudo update-alternatives --config g++  --> set g++-14

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.8/lib64
export PATH=${PATH}:/usr/local/cuda-12.8/bin

## Set compilation and runtime environment for CUDA 12.8 with GCC-14 and G++-14
sudo update-alternatives --config gcc  --> set gcc-11
sudo update-alternatives --config g++  --> set g++-11

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11.7/lib64
export PATH=${PATH}:/usr/local/cuda-11.7/bin

## Compilation
nvcc -O3 -arch=sm_75 percentileMethods.cu -o percentileMethods
./percentileMethods
# Note that the flag sm_75 shall be adapted for the GPU, here it matches the RTX2080 Ti (Turing architecture)

## Objective
Compare 2 percentile methods:
 - https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
 - https://www.nvidia.com/content/GTC-2010/pdfs/2220_GTC2010.pdf

 Generate a volume of float of size 80 * 80 * 146 * 300 and pass it on both methods.
 Benchmark the memory consumed and the timing.

 ## To profile
 nsys profile --trace=cuda --cuda-memory-usage=true ./percentileMethods

 ## Results
 Stored in the NAS (https://10.37.0.103:5001/) here: /Shared/Alyson/Percentile