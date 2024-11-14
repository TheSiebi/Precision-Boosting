#include <assert.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../split.h"
#include "../cuda_utils.h"
#include "../timer.h"

__global__ void basic_mixed_precision_matmul(const half* A, const half* B, float *C, int M, int K, int N)
{
    const auto index = [](int row, int col, int rows, int cols)
    {
        return row * cols + col; // row-major
    };
    const int tid = blockIdx.x;
    const int row = tid / N;
    const int col = tid % N;
    C[index(row, col, M, N)] = 0.f;
    // TOTAL: 2*K flops16
    // SHOULD BE flops16 to emulate tensor cores, will be changed in main
    for (int l = 0; l < K; ++l)
        // C[index(row, col, M, N)] += (float)(A[index(row, l, M, K)] * B[index(l, col, K, N)]);
        // 2 flops16 (see note above why 16)
        C[index(row, col, M, N)] += __half2float(A[index(row, l, M, K)]) * __half2float(B[index(l, col, K, N)]);
}

flop_counts matmul_simpleOotomo_v0(float *A, float *B, float *C, int M, int K, int N)
{
    // Allocate host memory
    half* A16      = (half*) malloc(M * K * sizeof(half));
    half* dA16     = (half*) malloc(M * K * sizeof(half));
    half* B16      = (half*) malloc(K * N * sizeof(half));
    half* dB16     = (half*) malloc(K * N * sizeof(half));
    float* A16B16  = (float*)malloc(M * N * sizeof(float));
    float* dA16B16 = (float*)malloc(M * N * sizeof(float));
    float* A16dB16 = (float*)malloc(M * N * sizeof(float));

    // Split (host)
    // 2*M*K flops32
    splitf_Ootomo_v0(A, A16, dA16, M, K);
    // 2*K*N flops32
    splitf_Ootomo_v0(B, B16, dB16, K, N);

    // Allocate device memory
    half*  dev_A16;
    half*  dev_dA16;
    half*  dev_B16;
    half*  dev_dB16;
    float* dev_A16B16;
    float* dev_dA16B16;
    float* dev_A16dB16;
    PRINT_ON_ERROR(cudaMalloc((void**)&dev_A16,     M * K * sizeof(half)));
    PRINT_ON_ERROR(cudaMalloc((void**)&dev_dA16,    M * K * sizeof(half)));
    PRINT_ON_ERROR(cudaMalloc((void**)&dev_B16,     K * N * sizeof(half)));
    PRINT_ON_ERROR(cudaMalloc((void**)&dev_dB16,    K * N * sizeof(half)));
    PRINT_ON_ERROR(cudaMalloc((void**)&dev_A16B16,  M * N * sizeof(float)));
    PRINT_ON_ERROR(cudaMalloc((void**)&dev_dA16B16, M * N * sizeof(float)));
    PRINT_ON_ERROR(cudaMalloc((void**)&dev_A16dB16, M * N * sizeof(float)));

    // Copy from host to device
    PRINT_ON_ERROR(cudaMemcpy(dev_A16,     A16,     M * K * sizeof(half), cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(dev_dA16,    dA16,    M * K * sizeof(half), cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(dev_B16,     B16,     K * N * sizeof(half), cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(dev_dB16,    dB16,    K * N * sizeof(half), cudaMemcpyHostToDevice));

    // Multiply matrices
    // 2*M*K*N flops16
    basic_mixed_precision_matmul<<<M * N, 1>>>(dev_A16, dev_B16, dev_A16B16, M, K, N);
    PRINT_ON_ERROR(cudaGetLastError());
    // 2*M*K*N flops16
    basic_mixed_precision_matmul<<<M * N, 1>>>(dev_dA16, dev_B16, dev_dA16B16, M, K, N);
    PRINT_ON_ERROR(cudaGetLastError());
    // 2*M*K*N flops16
    basic_mixed_precision_matmul<<<M * N, 1>>>(dev_A16, dev_dB16, dev_A16dB16, M, K, N);
    PRINT_ON_ERROR(cudaGetLastError());

    // Copy from device to host
    PRINT_ON_ERROR(cudaDeviceSynchronize());
    PRINT_ON_ERROR(cudaMemcpy(A16B16, dev_A16B16, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    PRINT_ON_ERROR(cudaMemcpy(dA16B16, dev_dA16B16, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    PRINT_ON_ERROR(cudaMemcpy(A16dB16, dev_A16dB16, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Accumulate (host)
    // M*N*3 flops32
    for (int i = 0; i < M * N; ++i)
    {
        const float ab = A16B16[i];
        const float dab = dA16B16[i];
        const float adb = A16dB16[i];
        C[i] = ab + (dab + adb) / 2048.f;
    }

    // Free device memory
    PRINT_ON_ERROR(cudaFree(dev_A16));
    PRINT_ON_ERROR(cudaFree(dev_dA16));
    PRINT_ON_ERROR(cudaFree(dev_B16));
    PRINT_ON_ERROR(cudaFree(dev_dB16));
    PRINT_ON_ERROR(cudaFree(dev_A16B16));
    PRINT_ON_ERROR(cudaFree(dev_dA16B16));
    PRINT_ON_ERROR(cudaFree(dev_A16dB16));

    // Free host memory
    free(A16);
    free(dA16);
    free(B16);
    free(dB16);
    free(A16B16);
    free(dA16B16);
    free(A16dB16);


    /* 
    TOTAL FLOP COUNTS:
    flops16:

    flops32:
    2*M*K
    + 2*K*N
    + M*N*3
     
    flops64:
    3*(2*M*K*N)
    */
    flop_counts counts = {0L, 0L, 0L};
    return counts;
}
