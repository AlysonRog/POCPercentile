#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>

#define CHECK_CUDA(expr)                                                     \
  do {                                                                       \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess) {                                               \
      std::cerr << "CUDA error " << cudaGetErrorString(_err)                 \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;       \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)


/**
 * @brief Computes the linear interpolation between 2 input values
 *
 * @param a the first value
 * @param b the second value
 * @param t interpolation parameter (between 0 and 1)
 *
 * @return the interpolated value
 */
inline float lerp_two(float a, float b, double t)
{
	return fmaf(static_cast<float>(t), (b - a), a);
}

/**
 * @brief Sort the input buffer dopplerVolume and computes the percentile with Thrust
 *        see: https://nvidia.github.io/cccl/thrust/api/function_group__sorting_1gad39e37d88f8334cbdd3e047a53e5cfba.html
 *
 * @param dopplerVolume the input buffer (memory on the host)
 * @param percentil the percentile rank
 *
 * @return the percentile score
 */
float percentileWithThrust(const std::vector<float>& dopplerVolume,
                           float percentil)
{
  unsigned int volumeSize = dopplerVolume.size();

	thrust::device_vector<float> d_dopplerVolume(dopplerVolume);

	const double r = percentil * (volumeSize - 1);
	const size_t k = static_cast<size_t>(floor(r));
	const double frac = r - static_cast<double>(k);

	thrust::sort(thrust::cuda::par.on(0), d_dopplerVolume.begin(), d_dopplerVolume.end()); // execute on stream 0
	float vk = d_dopplerVolume[k];
	float vk1 = (k + 1 < volumeSize) ? d_dopplerVolume[k + 1] : vk;

	const float res = (frac == 0.0 || k + 1 == volumeSize) ? vk : lerp_two(vk, vk1, frac);

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  return res;
}

/**
 * @brief Sort the input buffer dopplerVolume and computes the percentile with Radix
 *        see: https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
 *
 * @param dopplerVolume the input buffer (memory on the host)
 * @param percentil the percentile rank
 *
 * @return the percentile score
 */
float percentileWithDeviceRadixSort(const std::vector<float>& dopplerVolume, float percentil)
{
  unsigned int volumeSize = dopplerVolume.size();
  float *d_in = nullptr;
  float *d_out = nullptr;

  CHECK_CUDA(cudaMalloc(&d_in, volumeSize * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, volumeSize * sizeof(float)));

  // Copy input data
  CHECK_CUDA(cudaMemcpy(d_in, dopplerVolume.data(),
                        volumeSize * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Temporary storage
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // Request temp storage size
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, volumeSize);

  // Allocate temp storage
  CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run sorting
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, volumeSize);

	const double r = percentil * (volumeSize - 1);
	const size_t k = static_cast<size_t>(floor(r));
	const double frac = r - static_cast<double>(k);

  // Retrieve the k-nth and the k-nth+1 values from the device.
  // It is more efficient to retrieve just those 2 values from the sorted buffer rather than copying the
  // full sorted buffer on the host to retrieve those values.
  // I guess this mimics the behavior of the thrust::device_vector accessors.
  float valueK;
  float valueK1;

  CHECK_CUDA(cudaMemcpy(&valueK, d_out + k, sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&valueK1, d_out + k + 1, sizeof(float), cudaMemcpyDeviceToHost));

	float vk = valueK;
	float vk1 = (k + 1 < volumeSize) ? valueK1 : vk;

	const float res = (frac == 0.0 || k + 1 == volumeSize) ? vk : lerp_two(vk, vk1, frac);

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Cleanup
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_temp_storage);

  return res;
}

/**
 * @brief Program to test the timing of the computation of the percentile using the sorting method of
 *        both Thrust and Radix for comparison.
 */
int main() {

    size_t freeB, totalB;
    CHECK_CUDA(cudaMemGetInfo(&freeB, &totalB));
    std::fprintf(stderr, "GPU free %.2f GiB / %.2f GiB\n",
             freeB / (1024.0*1024*1024), totalB / (1024.0*1024*1024));

    // Generate noise on the host
    // const unsigned int volumeSize = 80 * 80 * 146 * 300; // For real size test
    const unsigned int volumeSize = 80; // For quick test
    std::vector<float> h_noise(volumeSize);

    std::mt19937 rng(12345); // deterministic seed
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (unsigned int i = 0; i < volumeSize; ++i) {
        // Simple white noise in [0,1); replace with any noise you like
        h_noise[i] = dist(rng);
        //std::cout << "i :" << i << " value: " << h_noise[i] << std::endl;
    }

    float percentileRank {0.9f}; // Must be between 0 and 1

    std::cout << "Method Thrust: " << percentileWithThrust(h_noise, percentileRank) << std::endl;
    std::cout << "Method Radix:  " << percentileWithDeviceRadixSort(h_noise, percentileRank) << std::endl;

    return 0;
}