nclude <stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <iostream>



#define BLOCK_NUM 8   //块数量
#define THREAD_NUM 128 // 每个块中的线程数
#define R_SIZE BLOCK_NUM * THREAD_NUM
#define M_SIZE R_SIZE * R_SIZE

__global__ void mat_mul(int *mat1, int *mat2, int *result) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    // 每个线程计算一行
    const int gid = bid * THREAD_NUM + tid;
    const int row = gid / R_SIZE;
    const int col = gid % R_SIZE;
    for (int n = 0; n < R_SIZE; n++) {
        result[row*R_SIZE+col] += mat1[row*R_SIZE+n] * mat2[n*R_SIZE+col];
    }

}

int main(int argc, char *argv[]) {
    int *mat1, *mat2, *result;
    int *g_mat1, *g_mat2, *g_mat_result;

    // 用一位数组表示二维矩阵
    mat1 = (int*) malloc(M_SIZE * sizeof(int));
    mat2 = (int*) malloc(M_SIZE * sizeof(int));
    result = (int*) malloc(M_SIZE * sizeof(int));

    // initialize
    for (int i = 0; i < M_SIZE; i++) {
        mat1[i] = rand()/1000000;
        mat2[i] = rand()/1000000;
        result[i] = 0;

    }

    cudaMalloc((void **)&g_mat1, sizeof(int) * M_SIZE);
    cudaMalloc((void **)&g_mat2, sizeof(int) * M_SIZE);
    cudaMalloc((void **)&g_mat_result, sizeof(int) * M_SIZE);

    cudaMemcpy(g_mat1, mat1, sizeof(int) * M_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(g_mat2, mat2, sizeof(int) * M_SIZE, cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
    float time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    mat_mul<<<BLOCK_NUM, THREAD_NUM>>>(g_mat1, g_mat2, g_mat_result);

        cudaEventRecord( stop, 0  );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    std::cout<<"cost time is:"<<time<<std::endl;
    cudaEventDestroy( start );
    cudaEventDestroy( stop );


    cudaMemcpy(result, g_mat_result, sizeof(int) * M_SIZE, cudaMemcpyDeviceToHost);
}

