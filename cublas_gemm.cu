#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <iostream>

void print_matrix(float* mat, int row, int col) {
    for (int i = 0; i < row * col; i++) {
        printf("%f\t", mat[i]);
        if ((i+1) % col == 0) {
            printf("\n");
        }

    }
     printf("----------------------------\n");
}


void cpu_mm(float *mat1, float *mat2, float* result){
        int m_size  = 32*256;
        for (int r = 0; r < 1; r++) {
        for (int c = 0; c < 10; c++) {
            for (int n = 0; n < m_size; n++) {
                result[r*m_size + c] += mat1[r*m_size+n] * mat2[n*m_size+c];
            }
        }

        }
}

 int check(float *c_result, float *g_result) {
     for(int i=0; i<10; i++)
         if(c_result[i] != g_result[i]){
             std::cout<<"check failed, original is:"<<c_result[i]<<"result is:"<<g_result[i]<<std::endl;
         }

 }


int main(int argc, char *argv[]) {
    float *mat1, *mat2, *result;
    float *g_mat1, *g_mat2, *g_mat_result;
    int r_size, m_size;    // 矩阵行数，矩阵size

    cudaError_t cudaStat;
    cublasHandle_t handle;
    cublasStatus_t stat;

    if (argc > 1) {
        r_size = atoi(argv[1]);
    } else {
        r_size = 8192;
    }
    m_size = r_size * r_size;

    // 用一位数组表示二维矩阵
    mat1 = (float*) malloc(m_size * sizeof(float));
    mat2 = (float*) malloc(m_size * sizeof(float));
    result = (float*) malloc(m_size * sizeof(float));

    // initialize
    for (int i = 0; i < m_size; i++) {
        mat1[i] = rand()/10000000;
        mat2[i] = rand()/10000000;
        result[i] = 0;
    }

    cudaStat = cudaMalloc((void **)&g_mat1, sizeof(*mat1) * m_size);
    cudaStat = cudaMalloc((void **)&g_mat2, sizeof(*mat2) * m_size);
    cudaStat = cudaMalloc((void **)&g_mat_result, sizeof(*result) * m_size);
    printf("cudaStat %d\n", cudaStat);

    // initialize CUBLAS context
    stat = cublasCreate(&handle);

    stat = cublasSetMatrix(r_size, r_size, sizeof(*mat1), mat1, r_size, g_mat1, r_size);
    stat = cublasSetMatrix(r_size, r_size, sizeof(*mat2), mat2, r_size, g_mat2, r_size);
    stat = cublasSetMatrix(r_size, r_size, sizeof(*result), result, r_size, g_mat_result, r_size);

    float al = 1.0f;
    float bet = 0.0f;

         cudaEvent_t start, stop;
     float time;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     cudaEventRecord( start, 0 );

    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        r_size, r_size, r_size, &al, g_mat1,
        r_size, g_mat2, r_size, &bet, g_mat_result, r_size);

        cudaEventRecord( stop, 0  );
    cudaEventSynchronize( start );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    std::cout<<"cost time is:"<<time<<std::endl;
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    stat = cublasGetMatrix(r_size, r_size, sizeof(*result), g_mat_result, r_size, result, r_size);
    printf("cublas %d\n",stat);
    // cudaMemcpy(result, g_mat_result, sizeof(float) * m_size, cudaMemcpyDeviceToHost);

        float* c_result = (float*) malloc(m_size * sizeof(float));
        cpu_mm(mat1, mat2, c_result);
        check(c_result, result);
    if (r_size < 10) {
        printf("-----mat1----\n");
        print_matrix(mat1, r_size, r_size);
        printf("-----mat2----\n");
        print_matrix(mat2, r_size, r_size);
        printf("----mat1 * mat2---\n");
        print_matrix(result, r_size, r_size);
    }
    cudaFree(g_mat1);
    cudaFree(g_mat2);
    cudaFree(g_mat_result);
    free(mat1);
    free(mat2);
    free(result);
}

