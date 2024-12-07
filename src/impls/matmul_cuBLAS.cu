#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <assert.h>

#include "../matmul.h"
#include "../profiler.h"
#include "../cuda_utils.h"
#include "../timer.h"

flop_counts matmul_cuBLAS32(float *h_A, float *h_B, float *h_C, size_t M, size_t K, size_t N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed. %s: %s\n",
                cublasGetStatusName(status), cublasGetStatusString(status));
        flop_counts counts = {0L, 0L, 0L};
        return counts;
    }

    PROFILE_FUNCTION_SEGMENT_START("allocate");
    float *d_A, *d_B, *d_C;
    PRINT_ON_ERROR(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    PRINT_ON_ERROR(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    PRINT_ON_ERROR(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    // Copy data from host to device
    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    PRINT_ON_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Perform matrix multiplication
    PROFILE_SEGMENTS_SWITCH("matmul");
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, M, K,
                         &alpha,
                         d_B, N,
                         d_A, K,
                         &beta,
                         d_C, N);

    CUDA_DEVICE_SYNCHRONIZE();
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS multiplication failed\n");
    }

    // Copy the result back to host
    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    PRINT_ON_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    PROFILE_SEGMENTS_SWITCH("free");
    PRINT_ON_ERROR(cudaFree(d_A));
    PRINT_ON_ERROR(cudaFree(d_B));
    PRINT_ON_ERROR(cudaFree(d_C));
    cublasDestroy(handle);
    PROFILE_SEGMENT_FUNCTION_END();

    flop_counts counts = {0L, 2L*M*K*N, 0L};
    return counts;
}

flop_counts matmul_cuBLAS64(double *h_A, double *h_B, double *h_C, size_t M, size_t K, size_t N) {
    const double alpha = 1.0f;
    const double beta = 0.0f;

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed. %s: %s\n",
                cublasGetStatusName(status), cublasGetStatusString(status));
        flop_counts counts = {0L, 0L, 0L};
        return counts;
    }

    PROFILE_FUNCTION_SEGMENT_START("allocate");
    double *d_A, *d_B, *d_C;
    PRINT_ON_ERROR(cudaMalloc((void**)&d_A, M * K * sizeof(double)));
    PRINT_ON_ERROR(cudaMalloc((void**)&d_B, K * N * sizeof(double)));
    PRINT_ON_ERROR(cudaMalloc((void**)&d_C, M * N * sizeof(double)));

    // Copy data from host to device
    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    PRINT_ON_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(double), cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(double), cudaMemcpyHostToDevice));

    // Perform matrix multiplication
    PROFILE_SEGMENTS_SWITCH("matmul");
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, M, K,
                         &alpha,
                         d_B, N,
                         d_A, K,
                         &beta,
                         d_C, N);

    CUDA_DEVICE_SYNCHRONIZE();
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS multiplication failed\n");
    }

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    // Copy the result back to host
    PRINT_ON_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));

    PROFILE_SEGMENTS_SWITCH("free");
    // Free device memory
    PRINT_ON_ERROR(cudaFree(d_A));
    PRINT_ON_ERROR(cudaFree(d_B));
    PRINT_ON_ERROR(cudaFree(d_C));
    cublasDestroy(handle);
    PROFILE_SEGMENT_FUNCTION_END();

    flop_counts counts = {0L, 0L, 2L*M*K*N};
    return counts;
}
