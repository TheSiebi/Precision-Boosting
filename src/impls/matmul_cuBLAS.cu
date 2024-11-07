#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <assert.h>

#include "../matmul.h"
#include "../profiler.h"

void matmul_cuBLAS32(float *h_A, float *h_B, float *h_C, int M, int K, int N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed. %s: %s\n",
                cublasGetStatusName(status), cublasGetStatusString(status));
        return;
    }

    PROFILE_FUNCTION_SEGMENT_START("allocate");
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Copy data from host to device
    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication
    PROFILE_SEGMENTS_SWITCH("matmul");
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, M, K,
                         &alpha,
                         d_B, N,
                         d_A, K,
                         &beta,
                         d_C, N);

    cudaDeviceSynchronize();
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS multiplication failed\n");
    }

    // Copy the result back to host
    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    PROFILE_SEGMENTS_SWITCH("free");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

void matmul_cuBLAS64(double *h_A, double *h_B, double *h_C, int M, int K, int N) {
    const double alpha = 1.0f;
    const double beta = 0.0f;

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed. %s: %s\n",
                cublasGetStatusName(status), cublasGetStatusString(status));
        return;
    }

    PROFILE_FUNCTION_SEGMENT_START("allocate");
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(double));
    cudaMalloc((void**)&d_B, K * N * sizeof(double));
    cudaMalloc((void**)&d_C, M * N * sizeof(double));

    // Copy data from host to device
    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    cudaMemcpy(d_A, h_A, M * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(double), cudaMemcpyHostToDevice);

    // Perform matrix multiplication
    PROFILE_SEGMENTS_SWITCH("matmul");
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, M, K,
                         &alpha,
                         d_B, N,
                         d_A, K,
                         &beta,
                         d_C, N);

    cudaDeviceSynchronize();
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS multiplication failed\n");
    }

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    // Copy the result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    PROFILE_SEGMENTS_SWITCH("free");
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    PROFILE_SEGMENT_FUNCTION_END();
}
