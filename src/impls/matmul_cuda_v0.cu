#include <assert.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "../matmul.h"
#include "../profiler.h"
#include "../cuda_utils.h"
#include "../timer.h"

__global__ void matmul_cuda_v0_kernel(double *A, double *B, double *C, int M, int K, int N) 
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    double result = 0.0;
    for (int k = 0; k < K; k++) 
    {
        result += A[m*K + k] * B[k*N + n];
    }
    C[m*N + n] = result;
}


flop_counts matmul_cuda_v0(double *A, double *B, double *C, int M, int K, int N) 
{
    assert((M & 0xF) == 0);
    assert((K & 0xF) == 0);
    assert((N & 0xF) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate");
    size_t ASize = M * K * sizeof(double);
    size_t BSize = K * N * sizeof(double);
    size_t CSize = M * N * sizeof(double);

    double *deviceA, *deviceB, *deviceC;
    PRINT_ON_ERROR(cudaMalloc(&deviceA, ASize));
    PRINT_ON_ERROR(cudaMalloc(&deviceB, BSize));
    PRINT_ON_ERROR(cudaMalloc(&deviceC, CSize));

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    PRINT_ON_ERROR(cudaMemcpy(deviceA, A, ASize, cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(deviceB, B, BSize, cudaMemcpyHostToDevice));

    PROFILE_SEGMENTS_SWITCH("matmul");
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(M/threadsPerBlock.x, N/threadsPerBlock.y);
    matmul_cuda_v0_kernel<<<blocks, threadsPerBlock>>>(deviceA, deviceB, deviceC, M, K, N);
    PRINT_ON_ERROR(cudaGetLastError());

    PRINT_ON_ERROR(cudaDeviceSynchronize());

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    PRINT_ON_ERROR(cudaMemcpy(C, deviceC, CSize, cudaMemcpyDeviceToHost));

    PROFILE_SEGMENTS_SWITCH("free");
    PRINT_ON_ERROR(cudaFree(deviceA));
    PRINT_ON_ERROR(cudaFree(deviceB));
    PRINT_ON_ERROR(cudaFree(deviceC));
    PROFILE_SEGMENT_FUNCTION_END();

    flop_counts counts = {0L, 0L, 0L};
    return counts;
}

