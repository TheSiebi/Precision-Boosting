#include <assert.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../cuda_utils.h"
#include "../matmul.h"
#include "../profiler.h"

#include "../timer.h"

#include "matmul_cuda.h"

__global__ void split_cuda(float *A, half *A0, half *A1, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < N)
    {
        float value = A[i];
        half mainPart = (half)value;
        A0[i] = mainPart;
        A1[i] = (half)(value - (float)mainPart);
    }
}

template<int version>
void matmul_simpleMarkidis(float *A, float *B, float *C, int M, int K, int N) 
{
    assert((M % 16) == 0);
    assert((K % 16) == 0);
    assert((N % 16) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate cpu");

    size_t ASize = M * K * sizeof(half);
    size_t BSize = K * N * sizeof(half);
    size_t CSize = M * N * sizeof(float);
    
    float *hostC[4];
    for(int i = 0; i < 4; i++)
        hostC[i] = (float*)malloc(CSize);


    PROFILE_SEGMENTS_SWITCH("allocate gpu");

    half *deviceA[2], *deviceB[2];
    float *deviceC[4];
    float *deviceAFull, *deviceBFull;
    for(int i = 0; i < 2; i++)
    {
        cudaGetLastError();
        PRINT_ON_ERROR(cudaMalloc(&deviceA[i], ASize));
        PRINT_ON_ERROR(cudaMalloc(&deviceB[i], BSize));
    }
    for(int i = 0; i < 4; i++)
        PRINT_ON_ERROR(cudaMalloc(&deviceC[i], CSize));
    PRINT_ON_ERROR(cudaMalloc(&deviceAFull, M*K*sizeof(float)));
    PRINT_ON_ERROR(cudaMalloc(&deviceBFull, K*N*sizeof(float)));

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");

    PRINT_ON_ERROR(cudaMemcpy(deviceAFull, A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(deviceBFull, B, K*N*sizeof(float), cudaMemcpyHostToDevice));

    PROFILE_SEGMENTS_SWITCH("split");

    split_cuda<<<DivRoundUp(M*K, 256), 256>>>(deviceAFull, deviceA[0], deviceA[1], M * K);
    PRINT_ON_ERROR(cudaGetLastError());
    split_cuda<<<DivRoundUp(K*N, 256), 256>>>(deviceBFull, deviceB[0], deviceB[1], K * N);
    PRINT_ON_ERROR(cudaGetLastError());

    PRINT_ON_ERROR(cudaDeviceSynchronize());

    PROFILE_SEGMENTS_SWITCH("matmul");
    for(int i = 0; i < 4; i++)
    {
        matmul<half, float, version>(deviceA[i/2], deviceB[i%2], deviceC[i], M, K, N);
    }
    PRINT_ON_ERROR(cudaDeviceSynchronize());

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");

    for(int i = 0; i < 4; i++)
        PRINT_ON_ERROR(cudaMemcpy(hostC[i], deviceC[i], CSize, cudaMemcpyDeviceToHost));

    PROFILE_SEGMENTS_SWITCH("merge");

    for(int i = 0; i < M * N; i++)
        C[i] = hostC[0][i] + hostC[1][i] + hostC[2][i] + hostC[3][i];

    PROFILE_SEGMENTS_SWITCH("free");

    for(int i = 0; i < 2; i++)
    {
        PRINT_ON_ERROR(cudaFree(deviceA[i]));
        PRINT_ON_ERROR(cudaFree(deviceB[i]));
    }
    for(int i = 0; i < 4; i++)
        PRINT_ON_ERROR(cudaFree(deviceC[i]));
    PRINT_ON_ERROR(cudaFree(deviceAFull));
    PRINT_ON_ERROR(cudaFree(deviceBFull));

    PROFILE_SEGMENT_FUNCTION_END();
}

/**
 * Flop counts of markidis should be very similar to Ootomo, with the difference that we
 * only require one flop32 for splitting an element and similarly for merging.
 * Furthermore, we perform 4 fp16 matmuls instead of 3
 * 
 * flops16:
 * 4*(2*M*K*N) (4 matmuls)
 * 
 * flops32:
 * M*K + K*N (splitting A and B)
 * + 3*N*M (merging into C)
 */
flop_counts matmul_simpleMarkidis_v0(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_simpleMarkidis<0>(A, B, C, M, K, N);
    flop_counts counts = {8L*M*K*N, M*K + K*N + 3L*N*M, 0L};
    return counts;
}

flop_counts matmul_simpleMarkidis_v1(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_simpleMarkidis<1>(A, B, C, M, K, N);
    flop_counts counts = {8L*M*K*N, M*K + K*N + 3L*N*M, 0L};
    return counts;
}

flop_counts matmul_simpleMarkidis_v2(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_simpleMarkidis<2>(A, B, C, M, K, N);
    flop_counts counts = {8L*M*K*N, M*K + K*N + 3L*N*M, 0L};
    return counts;
}

flop_counts matmul_simpleMarkidis_v3(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_simpleMarkidis<3>(A, B, C, M, K, N);
    flop_counts counts = {8L*M*K*N, M*K + K*N + 3L*N*M, 0L};
    return counts;
}

flop_counts matmul_simpleMarkidis_v4(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_simpleMarkidis<4>(A, B, C, M, K, N);
    flop_counts counts = {8L*M*K*N, M*K + K*N + 3L*N*M, 0L};
    return counts;
}

