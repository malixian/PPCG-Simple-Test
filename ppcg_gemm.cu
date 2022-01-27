#include "cuda.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

__global__ void kernel0(float *a, float *b, float *c, int N)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_a[32][32];
    __shared__ float shared_b[32][32];
    float private_c[1][2];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < N; c0 += 8192)
      for (int c1 = 32 * b1; c1 < N; c1 += 8192) {
        if (N >= t0 + c0 + 1 && N >= t1 + c1 + 1) {
          private_c[0][0] = c[(t0 + c0) * N + (t1 + c1)];
          if (N >= t1 + c1 + 17)
            private_c[0][1] = c[(t0 + c0) * N + (t1 + c1 + 16)];
        }
        for (int c2 = 0; c2 < N; c2 += 32) {
          if (N >= t0 + c0 + 1)
            for (int c4 = t1; c4 <= ppcg_min(31, N - c2 - 1); c4 += 16)
              shared_a[t0][c4] = a[(t0 + c0) * N + (c2 + c4)];
          if (N >= t0 + c2 + 1)
            for (int c4 = t1; c4 <= ppcg_min(31, N - c1 - 1); c4 += 16)
              shared_b[t0][c4] = b[(t0 + c2) * N + (c1 + c4)];
          __syncthreads();
          if (N >= t0 + c0 + 1 && N >= t1 + c1 + 1)
            for (int c3 = 0; c3 <= ppcg_min(31, N - c2 - 1); c3 += 1) {
              private_c[0][0] += (shared_a[t0][c3] * shared_b[c3][t1]);
              if (N >= t1 + c1 + 17)
                private_c[0][1] += (shared_a[t0][c3] * shared_b[c3][t1 + 16]);
            }
          __syncthreads();
        }
        if (N >= t0 + c0 + 1 && N >= t1 + c1 + 1) {
          c[(t0 + c0) * N + (t1 + c1)] = private_c[0][0];
          if (N >= t1 + c1 + 17)
            c[(t0 + c0) * N + (t1 + c1 + 16)] = private_c[0][1];
        }
        __syncthreads();
      }
}

int main()
{
    int N = 256 * 32;
    int M_SIZE = N*N;
    float* a = (float*) malloc(M_SIZE * sizeof(float));
    float* b = (float*) malloc(M_SIZE * sizeof(float));
    float* c = (float*) malloc(M_SIZE * sizeof(float));

    for (int i = 0; i < M_SIZE; ++i){
        a[i] = rand()/1000000;
        b[i] = rand()/1000000;
        c[i] = 0.0;
    }


        #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
        if (N >= 1) {
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

          float *dev_a;
          float *dev_b;
          float *dev_c;

          cudaCheckReturn(cudaMalloc((void **) &dev_a, (N) * (N) * sizeof(float)));
          cudaCheckReturn(cudaMalloc((void **) &dev_b, (N) * (N) * sizeof(float)));
          cudaCheckReturn(cudaMalloc((void **) &dev_c, (N) * (N) * sizeof(float)));

          cudaCheckReturn(cudaMemcpy(dev_a, a, (N) * (N) * sizeof(float), cudaMemcpyHostToDevice));
          cudaCheckReturn(cudaMemcpy(dev_b, b, (N) * (N) * sizeof(float), cudaMemcpyHostToDevice));
          cudaCheckReturn(cudaMemcpy(dev_c, c, (N) * (N) * sizeof(float), cudaMemcpyHostToDevice));
          {
            dim3 k0_dimBlock(16, 32);
            dim3 k0_dimGrid(ppcg_min(256, (N + 31) / 32), ppcg_min(256, (N + 31) / 32));

                cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );

            kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_a, dev_b, dev_c, N);

                cudaEventRecord( stop, 0  );

        cudaEventSynchronize( start );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &time, start, stop );
        printf("cost time is %.3f", time);
        std::cout<<"cost time is:"<<time<<std::endl;
        cudaEventDestroy( start );
        cudaEventDestroy( stop );

            cudaCheckKernel();
          }

          cudaCheckReturn(cudaMemcpy(c, dev_c, (N) * (N) * sizeof(float), cudaMemcpyDeviceToHost));
          cudaCheckReturn(cudaFree(dev_a));
          cudaCheckReturn(cudaFree(dev_b));
          cudaCheckReturn(cudaFree(dev_c));
        }

        float sum = 0;
        for(int i=0; i<2; i++)
                sum += c[i];
        std::cout<<sum<<std::endl;

        return EXIT_SUCCESS;
}

