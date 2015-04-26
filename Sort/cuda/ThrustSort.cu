#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

#include <algorithm>
#include <time.h>
#include <limits.h>
#include <math.h>

bool thrustSort (int numElements, int numIterations) {

  thrust::host_vector<double> h_keys(numElements);
  thrust::host_vector<double> h_keysSorted(numElements);

  // Fill up with some random data
  thrust::default_random_engine rng(clock());
  thrust::uniform_real_distribution<double> u01(0, 1);

  for (int i = 0; i < (int)numElements; i++)
    h_keys[i] = u01(rng);

  // Copy data onto the GPU
  thrust::device_vector<double> d_keys = h_keys;

  // run multiple iterations to compute an average sort time
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

  float totalTime = 0;

  for (unsigned int i = 0; i < numIterations; i++)
  {
    // reset data before sort
    d_keys = h_keys;

    cudaEventRecord(start_event, 0);

    thrust::sort(d_keys.begin(), d_keys.end());

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    float time = 0;
    cudaEventElapsedTime(&time, start_event, stop_event);
    totalTime += time;
  }
    printf("Sorting %d elements\n. Average time of %d runs is: %.5f ms\n",
        numElements, numIterations, (totalTime) / numIterations);

  // Get results back to host for correctness checking
  thrust::copy(d_keys.begin(), d_keys.end(), h_keysSorted.begin());

  // Check results
  bool bTestResult = thrust::is_sorted(h_keysSorted.begin(), h_keysSorted.end());

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  return bTestResult;
}

int main() {
  bool bTestResult = false;
  for (int size = 10; size <= 30; size++)
  {
    int elems = pow(2,size);
    bTestResult = thrustSort(elems, 5);
    printf(bTestResult ? "Test passed\n" : "Test failed!\n");
  }

  return 0;
}
