#include <CL/sycl.hpp>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

void kernel0(float *a, float *b, float *c, int N, sycl::nd_item<3> item_ct1,
             sycl::accessor<float,2, sycl::access_mode::read_write, sycl::access::target::local> shared_a,
             sycl::accessor<float,2, sycl::access_mode::read_write, sycl::access::target::local> shared_b)
{
    int b0 = item_ct1.get_group(1), b1 = item_ct1.get_group(2);
    int t0 = item_ct1.get_local_id(1), t1 = item_ct1.get_local_id(2);

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
          /*
 *           DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
 *                     sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
 *                               better performance, if there is no access to global memory.
 *                                         */
          item_ct1.barrier();
          if (N >= t0 + c0 + 1 && N >= t1 + c1 + 1)
            for (int c3 = 0; c3 <= ppcg_min(31, N - c2 - 1); c3 += 1) {
              private_c[0][0] += (shared_a[t0][c3] * shared_b[c3][t1]);
              if (N >= t1 + c1 + 17)
                private_c[0][1] += (shared_a[t0][c3] * shared_b[c3][t1 + 16]);
            }
          /*
 *           DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
 *                     sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
 *                               better performance, if there is no access to global memory.
 *                                         */
          item_ct1.barrier();
        }
        if (N >= t0 + c0 + 1 && N >= t1 + c1 + 1) {
          c[(t0 + c0) * N + (t1 + c1)] = private_c[0][0];
          if (N >= t1 + c1 + 17)
            c[(t0 + c0) * N + (t1 + c1 + 16)] = private_c[0][1];
        }
        /*
 *         DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
 *                 sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
 *                         better performance, if there is no access to global memory.
 *                                 */
        item_ct1.barrier();
      }
}

int main() try {
    sycl::gpu_selector s;
    sycl::queue q_ct1(s) ;
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
/*
 * DPCT1001:3: The statement could not be removed.
 * */
/*
 * DPCT1000:4: Error handling if-stmt was detected but could not be rewritten.
 * */
/*
 * DPCT1009:6: SYCL uses exceptions to report errors and does not use the error
 * codes. The original code was commented out and a warning string was inserted.
 * You need to rewrite this code.
 * */
#define cudaCheckReturn(ret)                                                                   \
    do {                                                                                       \
        int cudaCheckReturn_e = (ret);                                                         \
        if (cudaCheckReturn_e != 0) {                                                          \
            fprintf(                                                                           \
                stderr, "CUDA error: %s\n",                                                    \
                "cudaGetErrorString not supported" /*cudaGetErrorString(cudaCheckReturn_e)*/); \
            fflush(stderr);                                                                    \
        }                                                                                      \
        assert(cudaCheckReturn_e == 0);                                                        \
    } while (0)
#define cudaCheckKernel()                                                      \
    do {                                                                       \
        cudaCheckReturn(0);                                                    \
    } while (0)

          float *dev_a;
          float *dev_b;
          float *dev_c;

          /*
 *           DPCT1003:5: Migrated API does not return error code. (*, 0) is
 *                     inserted. You may need to rewrite this code.
 *                               */
          cudaCheckReturn(
              (dev_a = sycl::malloc_device<float>((N) * (N), q_ct1), 0));
          /*
 *           DPCT1003:7: Migrated API does not return error code. (*, 0) is
 *                     inserted. You may need to rewrite this code.
 *                               */
          cudaCheckReturn(
              (dev_b = sycl::malloc_device<float>((N) * (N), q_ct1), 0));
          /*
 *           DPCT1003:8: Migrated API does not return error code. (*, 0) is
 *                     inserted. You may need to rewrite this code.
 *                               */
          cudaCheckReturn(
              (dev_c = sycl::malloc_device<float>((N) * (N), q_ct1), 0));

          /*
 *           DPCT1003:9: Migrated API does not return error code. (*, 0) is
 *                     inserted. You may need to rewrite this code.
 *                               */
          cudaCheckReturn(
              (q_ct1.memcpy(dev_a, a, (N) * (N) * sizeof(float)).wait(), 0));
          /*
 *           DPCT1003:10: Migrated API does not return error code. (*, 0) is
 *                     inserted. You may need to rewrite this code.
 *                               */
          cudaCheckReturn(
              (q_ct1.memcpy(dev_b, b, (N) * (N) * sizeof(float)).wait(), 0));
          /*
 *           DPCT1003:11: Migrated API does not return error code. (*, 0) is
 *                     inserted. You may need to rewrite this code.
 *                               */
          cudaCheckReturn(
              (q_ct1.memcpy(dev_c, c, (N) * (N) * sizeof(float)).wait(), 0));
          {
            sycl::range<3> k0_dimBlock(1, 32, 16);
            sycl::range<3> k0_dimGrid(1, ppcg_min(256, (N + 31) / 32),
                                      ppcg_min(256, (N + 31) / 32));

                sycl::event start, stop;
                std::chrono::time_point<std::chrono::steady_clock> start_ct1;
                std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
        float time;
        /*
 *         DPCT1026:12: The call to cudaEventCreate was removed, because this call
 *                 is redundant in DPC++.
 *                         */
        /*
 *         DPCT1026:13: The call to cudaEventCreate was removed, because this call
 *                 is redundant in DPC++.
 *                         */
        /*
 *         DPCT1012:14: Detected kernel execution time measurement pattern and
 *                 generated an initial code for time measurements in SYCL. You can change
 *                         the way time is measured depending on your goals.
 *                                 */
        start_ct1 = std::chrono::steady_clock::now();

            /*
 *             DPCT1049:15: The workgroup size passed to the SYCL kernel may exceed
 *                         the limit. To get the device limit, query
 *                                     info::device::max_work_group_size. Adjust the workgroup size if
 *                                                 needed.
 *                                                             */
            stop = q_ct1.submit([&](sycl::handler &cgh) {
                sycl::range<2> shared_a_range_ct1(32, 32);
                sycl::range<2> shared_b_range_ct1(32, 32);

                sycl::accessor<float, 2, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    shared_a_acc_ct1(shared_a_range_ct1, cgh);
                sycl::accessor<float, 2, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    shared_b_acc_ct1(shared_b_range_ct1, cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(k0_dimGrid * k0_dimBlock, k0_dimBlock),
                    [=](sycl::nd_item<3> item_ct1) {
                        kernel0(dev_a, dev_b, dev_c, N, item_ct1,
                                //sycl::accessor<float, 2, sycl::access_mode::read_write, sycl::access::target::local>(
                                    shared_a_acc_ct1, 
                                //sycl::accessor<float, 2, sycl::access_mode::read_write, sycl::access::target::local>(
                                    shared_b_acc_ct1);
                    });
            });

                /*
 *                 DPCT1012:16: Detected kernel execution time measurement pattern
 *                                 and generated an initial code for time measurements in SYCL. You
 *                                                 can change the way time is measured depending on your goals.
 *                                                                 */
                stop.wait();
                stop_ct1 = std::chrono::steady_clock::now();

        time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                   .count();
        std::cout<<"cost time is:"<<time<<std::endl;
        /*
 *         DPCT1026:17: The call to cudaEventDestroy was removed, because this call
 *                 is redundant in DPC++.
 *                         */
        /*
 *         DPCT1026:18: The call to cudaEventDestroy was removed, because this call
 *                 is redundant in DPC++.
 *                         */

            /*
 *             DPCT1010:19: SYCL uses exceptions to report errors and does not use
 *                         the error codes. The call was replaced with 0. You need to rewrite
 *                                     this code.
 *                                                 */
            cudaCheckKernel();
          }

          /*
 *           DPCT1003:20: Migrated API does not return error code. (*, 0) is
 *                     inserted. You may need to rewrite this code.
 *                               */
          cudaCheckReturn(
              (q_ct1.memcpy(c, dev_c, (N) * (N) * sizeof(float)).wait(), 0));
          /*
 *           DPCT1003:21: Migrated API does not return error code. (*, 0) is
 *                     inserted. You may need to rewrite this code.
 *                               */
          cudaCheckReturn((sycl::free(dev_a, q_ct1), 0));
          /*
 *           DPCT1003:22: Migrated API does not return error code. (*, 0) is
 *                     inserted. You may need to rewrite this code.
 *                               */
          cudaCheckReturn((sycl::free(dev_b, q_ct1), 0));
          /*
 *           DPCT1003:23: Migrated API does not return error code. (*, 0) is
 *                     inserted. You may need to rewrite this code.
 *                               */
          cudaCheckReturn((sycl::free(dev_c, q_ct1), 0));
        }

        float sum = 0;
        for(int i=0; i<2; i++)
                sum += c[i];
        std::cout<<sum<<std::endl;

        return EXIT_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
