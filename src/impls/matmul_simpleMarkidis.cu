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

static __global__ 
void split_cuda(float *A, half *A0, half *A1, int N)
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

static __global__ 
void merge_cuda(float *C0, float *C1, float *C2, float *C3, float *C, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < N)
        C[i] = C0[i] + (C1[i] + (C2[i] + C3[i]));
}


template<int version>
flop_counts matmul_simpleMarkidis(float *A, float *B, float *C, int M, int K, int N) 
{
    assert((M % 16) == 0);
    assert((K % 16) == 0);
    assert((N % 16) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate cpu");

    size_t ASize = M * K * sizeof(half);
    size_t BSize = K * N * sizeof(half);
    size_t CSize = M * N * sizeof(float);
    
    PROFILE_SEGMENTS_SWITCH("allocate gpu");

    half *deviceA[2], *deviceB[2];
    float *deviceC[4];
    float *deviceCMerged;
    float *deviceAFull, *deviceBFull;
    for(int i = 0; i < 2; i++)
    {
        cudaGetLastError();
        PRINT_ON_ERROR(cudaMalloc(&deviceA[i], ASize));
        PRINT_ON_ERROR(cudaMalloc(&deviceB[i], BSize));
    }
    for(int i = 0; i < 4; i++)
        PRINT_ON_ERROR(cudaMalloc(&deviceC[i], CSize));
    PRINT_ON_ERROR(cudaMalloc(&deviceCMerged, CSize));
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

    PROFILE_SEGMENTS_SWITCH("merge");
    merge_cuda<<<DivRoundUp(M*N, 256), 256>>>
              (deviceC[0], deviceC[1], deviceC[2], deviceC[3], deviceCMerged, M*N);
    PRINT_ON_ERROR(cudaGetLastError());
    PRINT_ON_ERROR(cudaDeviceSynchronize());

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    PRINT_ON_ERROR(cudaMemcpy(C, deviceCMerged, CSize, cudaMemcpyDeviceToHost));

    PROFILE_SEGMENTS_SWITCH("free");

    for(int i = 0; i < 2; i++)
    {
        PRINT_ON_ERROR(cudaFree(deviceA[i]));
        PRINT_ON_ERROR(cudaFree(deviceB[i]));
    }
    for(int i = 0; i < 4; i++)
        PRINT_ON_ERROR(cudaFree(deviceC[i]));
    PRINT_ON_ERROR(cudaFree(deviceCMerged));
    PRINT_ON_ERROR(cudaFree(deviceAFull));
    PRINT_ON_ERROR(cudaFree(deviceBFull));

    PROFILE_SEGMENT_FUNCTION_END();
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
    flop_counts counts = {8L*M*K*N, M*K + K*N + 3L*N*M, 0L};
    return counts;
}

template flop_counts matmul_simpleMarkidis<0>(float *A, float *B, float *C, int M, int K, int N);
template flop_counts matmul_simpleMarkidis<1>(float *A, float *B, float *C, int M, int K, int N);
template flop_counts matmul_simpleMarkidis<2>(float *A, float *B, float *C, int M, int K, int N);
template flop_counts matmul_simpleMarkidis<3>(float *A, float *B, float *C, int M, int K, int N);
template flop_counts matmul_simpleMarkidis<4>(float *A, float *B, float *C, int M, int K, int N);

template<int splitCount>
static __global__ 
void split_cuda_double(double *A, half *ASplit, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < N)
    {
        double residual = A[i];
        #pragma unroll
        for(int j = 0; j < splitCount; j++)
        {
            half mainPart = (half)residual;
            ASplit[j*N+i] = mainPart;
            residual -= (double)mainPart;
        }
    }
}

template<int splitCount>
static __global__ 
void merge_cuda_double(float *CSplit, double *C, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < N)
    {
        double result = 0;
        #pragma unroll
        for(int j = 0; j < splitCount; j++)
            result += (double)CSplit[j*N+i];
        C[i] = result;
    }
}

template<int splitCount, int mergeCount>
flop_counts matmul_simpleMarkidis_double(double *A, double *B, double *C, int M, int K, int N,
                                        std::pair<int, int> mergePattern[mergeCount]) 
{
    assert((M % 16) == 0);
    assert((K % 16) == 0);
    assert((N % 16) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate cpu");

    size_t ASizeH = M * K * sizeof(half);
    size_t ASizeD = M * K * sizeof(double);
    size_t BSizeH = K * N * sizeof(half);
    size_t BSizeD = K * N * sizeof(double);
    size_t CSizeF = M * N * sizeof(float);
    size_t CSizeD = M * N * sizeof(double);
    
    PROFILE_SEGMENTS_SWITCH("allocate gpu");

    half *deviceA, *deviceB;
    float *deviceC;
    double *deviceCMerged;
    double *deviceAFull, *deviceBFull;
    cudaGetLastError();
    PRINT_ON_ERROR(cudaMalloc(&deviceA, ASizeH * splitCount));
    PRINT_ON_ERROR(cudaMalloc(&deviceB, BSizeH * splitCount));
    PRINT_ON_ERROR(cudaMalloc(&deviceC, CSizeF * mergeCount));
    PRINT_ON_ERROR(cudaMalloc(&deviceCMerged, CSizeD));
    PRINT_ON_ERROR(cudaMalloc(&deviceAFull, ASizeD));
    PRINT_ON_ERROR(cudaMalloc(&deviceBFull, BSizeD));

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");

    PRINT_ON_ERROR(cudaMemcpy(deviceAFull, A, ASizeD, cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(deviceBFull, B, BSizeD, cudaMemcpyHostToDevice));

    PROFILE_SEGMENTS_SWITCH("split");

    split_cuda_double<splitCount><<<DivRoundUp(M*K, 256), 256>>>(deviceAFull, deviceA, M * K);
    PRINT_ON_ERROR(cudaGetLastError());
    split_cuda_double<splitCount><<<DivRoundUp(K*N, 256), 256>>>(deviceBFull, deviceB, K * N);
    PRINT_ON_ERROR(cudaGetLastError());

    PRINT_ON_ERROR(cudaDeviceSynchronize());

    PROFILE_SEGMENTS_SWITCH("matmul");
    for(int i = 0; i < mergeCount; i++)
    {
        int aIndex = mergePattern[i].first * M * K;
        int bIndex = mergePattern[i].second * K * N;
        int cIndex = i * M * N;
        matmul<half, float, 3>(deviceA + aIndex, deviceB + bIndex, deviceC + cIndex, M, K, N);
    }
    PRINT_ON_ERROR(cudaDeviceSynchronize());

    PROFILE_SEGMENTS_SWITCH("merge");
    merge_cuda_double<mergeCount><<<DivRoundUp(M*N, 256), 256>>>(deviceC, deviceCMerged, M*N);
    PRINT_ON_ERROR(cudaGetLastError());
    PRINT_ON_ERROR(cudaDeviceSynchronize());

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    PRINT_ON_ERROR(cudaMemcpy(C, deviceCMerged, CSizeD, cudaMemcpyDeviceToHost));

    PROFILE_SEGMENTS_SWITCH("free");

    PRINT_ON_ERROR(cudaFree(deviceA));
    PRINT_ON_ERROR(cudaFree(deviceB));
    PRINT_ON_ERROR(cudaFree(deviceC));
    PRINT_ON_ERROR(cudaFree(deviceCMerged));
    PRINT_ON_ERROR(cudaFree(deviceAFull));
    PRINT_ON_ERROR(cudaFree(deviceBFull));

    PROFILE_SEGMENT_FUNCTION_END();
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
    flop_counts counts = {8L*M*K*N, M*K + K*N + 3L*N*M, 0L};
    return counts;
}

template<>
flop_counts matmul_simpleMarkidis_double<0>(double *A, double *B, double *C, int M, int K, int N)
{
    std::pair<int, int> merges[] = {{2, 2}, {2, 1}, {1, 2}, {0, 2}, {1, 1}, {2, 0}, {0, 1}, {1, 0}, {0, 0}};
    return matmul_simpleMarkidis_double<3, 9>(A, B, C, M, K, N, merges);
}

template<>
flop_counts matmul_simpleMarkidis_double<1>(double *A, double *B, double *C, int M, int K, int N)
{
    std::pair<int, int> merges[16];
    for(int i = 0; i < 16; i++)
        merges[i] = {i/4, i%4};
    return matmul_simpleMarkidis_double<4, 16>(A, B, C, M, K, N, merges);
}
