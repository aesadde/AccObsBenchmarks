#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>

template<unsigned int blockSize>
__device__ void warpReduce(volatile double *sdata, unsigned int tid){
  if (blockSize >= 64) { sdata[tid] += sdata[tid + 32] ; }
  if (blockSize >= 32) { sdata[tid] += sdata[tid + 16] ; }
  if (blockSize >= 16) { sdata[tid] += sdata[tid + 8]; }
  if (blockSize >= 8) { sdata[tid] += sdata[tid + 4]; }
  if (blockSize >= 4) {sdata[tid] += sdata[tid + 2]; }
  if (blockSize >= 2) {sdata[tid] += sdata[tid + 1]; }
}
template <unsigned int blockSize> __global__ void reduce6( double *g_idata,
    double *g_odata, unsigned int n) {
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid ;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 0;

  while (i < n) {
    sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize;
  }
  __syncthreads();
  if ( blockSize >= 512) { if ( tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }

  if ( blockSize >= 256) { if ( tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }

  if ( blockSize >= 128) { if ( tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

  if (tid < 32) {
    warpReduce<blockSize>(sdata,tid);


  }
  __syncthreads();

  if (tid == 0) {g_odata[blockIdx.x] = sdata[0]; }
}


int main() {
  for (int t = 0; t <= 4; t++) {
    for (int iter = 1; iter <= 15; iter++) {
      for (int bl = 1; bl <= 4; bl*=2) {

        printf("Iter %d, Blocks %d",t,bl);
        int blocks = 32*bl;
        int threads = 1024;
        int arraySize = 3200*1024*iter;
        int smemSize = threads * sizeof(double);
        int arrayBytes = arraySize * sizeof(double);

        printf("=====\n");
        printf("Input Size = %d\n",arraySize);

        double *h_in, *h_out;     // host arrays
        h_in = (double*) malloc(arrayBytes);
        double *d_in, *d_out; //device arrays
        h_out = (double*) malloc(smemSize);

        for (int i = 0; i < threads; i++) h_out[i] = 0;

        double result = 0;
        for (int i = 0; i < arraySize; i++) {
          h_in[i] = i;
          result += i;
        }

        cudaEvent_t start, stop, startT, stopT;
        float time,full;
        cudaEventCreate(&start);
        cudaEventCreate(&startT);
        cudaEventCreate(&stop);
        cudaEventCreate(&stopT);

        //allocate memory on device and copy
        cudaEventRecord(startT,0);
        cudaMalloc((void**)&d_in, arrayBytes);
        cudaMalloc((void**)&d_out, smemSize);

        cudaMemcpy(d_in, h_in, arrayBytes, cudaMemcpyHostToDevice);
        cudaEventRecord(start,0);
        reduce6<512><<<blocks,threads,smemSize>>>(d_in,d_out,arraySize);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaMemcpy(h_out, d_out, smemSize, cudaMemcpyDeviceToHost);

        double res = 0;
        for (int i = 0; i < blocks;i++){
          res += h_out[i];
        }
        cudaEventRecord(stopT,0);
        cudaEventSynchronize(stopT);

        std::cout << "Device Result is: " << h_out[0] << std::endl;
        std::cout << "Host Result is: " << result << std::endl;
        std::cout << "Result is: " << res << std::endl;
        printf("Result correct? %s\n", res == result ? "true" : "false");
        cudaEventElapsedTime(&time, start, stop);
        cudaEventElapsedTime(&full, startT, stopT);
        printf ("Time for the kernel: %f ms\n", time);
        printf ("Time Full: %f ms\n", full);

        free(h_in);
        free(h_out);
        cudaFree(d_in);
        cudaFree(d_out);
      }
    }
  }
}
