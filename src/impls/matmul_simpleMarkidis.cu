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
#include "./split_merge_cuda.h"

#include "../timer.h"

#include "matmul_cuda.h"

template<int version, int streamCount, bool useScale>
flop_counts matmul_simpleMarkidis(float *A, float *B, float *C, size_t M, size_t K, size_t N) 
{
    assert((M % 16) == 0);
    assert((K % 16) == 0);
    assert((N % 16) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate cpu");

    cudaStream_t streams[streamCount];
    for(int i = 0; i < streamCount; i++)
        PRINT_ON_ERROR(cudaStreamCreate(&streams[i]));

    constexpr float scale = useScale ? (float) (1 << 11) : 1.0f;
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
        PRINT_ON_ERROR(cudaMalloc(&deviceA[i], ASize));
        PRINT_ON_ERROR(cudaMalloc(&deviceB[i], BSize));
    }
    for(int i = 0; i < 4; i++)
        PRINT_ON_ERROR(cudaMalloc(&deviceC[i], CSize));
    PRINT_ON_ERROR(cudaMalloc(&deviceCMerged, CSize));
    PRINT_ON_ERROR(cudaMalloc(&deviceAFull, M*K*sizeof(float)));
    PRINT_ON_ERROR(cudaMalloc(&deviceBFull, K*N*sizeof(float)));

    PROFILE_SEGMENTS_SWITCH("memcpy h2d & split");

    size_t copyCountA = (M*K)/streamCount;
    size_t copySizeA = copyCountA * sizeof(float);
    size_t copyCountB = (K*N)/streamCount;
    size_t copySizeB = copyCountB * sizeof(float);
    for(int i = 0; i < streamCount; i++)
    {
        size_t offsetA = copyCountA * i;
        size_t offsetB = copyCountB * i;
        PRINT_ON_ERROR(
                cudaMemcpyAsync(deviceAFull + offsetA, 
                                 A + offsetA, copySizeA, 
                                 cudaMemcpyHostToDevice, streams[i])
        );
        PRINT_ON_ERROR(
                cudaMemcpyAsync(deviceBFull + offsetB, 
                                 B + offsetB, copySizeB, 
                                cudaMemcpyHostToDevice, streams[i])
        );
        split_2<float, half>
               <<<DivRoundUp(copyCountA, 256), 256, 0, streams[i]>>>
               (deviceAFull + offsetA, deviceA[0] + offsetA, deviceA[1] + offsetA, scale);
        split_2<float, half>
               <<<DivRoundUp(copyCountB, 256), 256, 0, streams[i]>>>
               (deviceBFull + offsetB, deviceB[0] + offsetB, deviceB[1] + offsetB, scale);
    }

    PRINT_ON_ERROR(cudaGetLastError());
    CUDA_DEVICE_SYNCHRONIZE();

    PROFILE_SEGMENTS_SWITCH("matmul");
    for(int i = 0; i < 4; i++)
    {
        matmulTensorCores<half, float, version>(deviceA[i/2], deviceB[i%2], deviceC[i], M, K, N);
    }
    CUDA_DEVICE_SYNCHRONIZE();

    PROFILE_SEGMENTS_SWITCH("merge");
    merge_2<float, float, true><<<DivRoundUp(M*N, 256), 256>>>
              (deviceCMerged, deviceC[0], deviceC[1], deviceC[2], deviceC[3], scale);
    PRINT_ON_ERROR(cudaGetLastError());
    CUDA_DEVICE_SYNCHRONIZE();

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

    for(int i = 0; i < streamCount; i++)
        PRINT_ON_ERROR(cudaStreamDestroy(streams[i]));

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

template<typename Type>
static __global__ 
void divide_cuda(Type *C, int N, double scale)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < N)
        C[i] /= scale;
}

template flop_counts matmul_simpleMarkidis<0, 1, false>(float *A, float *B, float *C, size_t M, size_t K, size_t N);
template flop_counts matmul_simpleMarkidis<1, 1, false>(float *A, float *B, float *C, size_t M, size_t K, size_t N);
template flop_counts matmul_simpleMarkidis<2, 1, false>(float *A, float *B, float *C, size_t M, size_t K, size_t N);
template flop_counts matmul_simpleMarkidis<3, 1, false>(float *A, float *B, float *C, size_t M, size_t K, size_t N);
template flop_counts matmul_simpleMarkidis<3, 4, false>(float *A, float *B, float *C, size_t M, size_t K, size_t N);
template flop_counts matmul_simpleMarkidis<4, 1, false>(float *A, float *B, float *C, size_t M, size_t K, size_t N);
template flop_counts matmul_simpleMarkidis<4, 1, true>(float *A, float *B, float *C, size_t M, size_t K, size_t N);
template flop_counts matmul_simpleMarkidis<5, 1, false>(float *A, float *B, float *C, size_t M, size_t K, size_t N);
template flop_counts matmul_simpleMarkidis<5, 1, true>(float *A, float *B, float *C, size_t M, size_t K, size_t N);

template<int splitCount, int mergeCount, typename mulInputType, typename mulType, typename mulOutputType, bool useTensorCores>
flop_counts matmul_simpleMarkidis_double(double *A, double *B, double *C, size_t M, size_t K, size_t N,
                                        std::pair<int, int> mergePattern[mergeCount], double scale) 
{
    assert((M % 16) == 0);
    assert((K % 16) == 0);
    assert((N % 16) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate cpu");

    size_t ASizeH = M * K * sizeof(mulInputType);
    size_t ASizeD = M * K * sizeof(double);
    size_t BSizeH = K * N * sizeof(mulInputType);
    size_t BSizeD = K * N * sizeof(double);
    size_t CSizeF = M * N * sizeof(mulOutputType);
    size_t CSizeD = M * N * sizeof(double);
    
    PROFILE_SEGMENTS_SWITCH("allocate gpu");

    mulInputType *deviceA, *deviceB;
    mulOutputType *deviceC;
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

    split_cuda_double<splitCount, mulInputType><<<DivRoundUp(M*K, 256), 256>>>(deviceAFull, deviceA, M * K, scale);
    PRINT_ON_ERROR(cudaGetLastError());
    split_cuda_double<splitCount, mulInputType><<<DivRoundUp(K*N, 256), 256>>>(deviceBFull, deviceB, K * N, scale);
    PRINT_ON_ERROR(cudaGetLastError());

    CUDA_DEVICE_SYNCHRONIZE();

    PROFILE_SEGMENTS_SWITCH("matmul");
    for(int i = 0; i < mergeCount; i++)
    {
        size_t aIndex = mergePattern[i].first * M * K;
        size_t bIndex = mergePattern[i].second * K * N;
        size_t cIndex = i * M * N;
        if constexpr(useTensorCores)
            matmulTensorCores<mulInputType, mulOutputType, 2>(deviceA + aIndex, deviceB + bIndex, deviceC + cIndex, M, K, N);
        else 
            matmulCUDACores<mulInputType, mulType, mulOutputType, 1>(deviceA + aIndex, deviceB + bIndex, deviceC + cIndex, M, K, N);
        
        double factor = std::pow(scale, mergePattern[i].first) * std::pow(scale, mergePattern[i].second);
        if (factor > 1.0)
            divide_cuda<mulOutputType><<<DivRoundUp(M*N, 256), 256>>>(deviceC + cIndex, M*N, factor);
    }
    CUDA_DEVICE_SYNCHRONIZE();

    PROFILE_SEGMENTS_SWITCH("merge");
    merge_cuda_double<mergeCount, mulOutputType><<<DivRoundUp(M*N, 256), 256>>>(deviceC, deviceCMerged, M*N);
    PRINT_ON_ERROR(cudaGetLastError());
    CUDA_DEVICE_SYNCHRONIZE();

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
flop_counts matmul_simpleMarkidis_double<0>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    std::pair<int, int> merges[] = {{2, 2}, {2, 1}, {1, 2}, {0, 2}, {1, 1}, {2, 0}, {0, 1}, {1, 0}, {0, 0}};
    return matmul_simpleMarkidis_double<3, 9, half, float, float, true>(A, B, C, M, K, N, merges, 1.0);
}

template<>
flop_counts matmul_simpleMarkidis_double<1>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    std::pair<int, int> merges[16];
    for(int i = 0; i < 16; i++)
        merges[i] = {i/4, i%4};
    return matmul_simpleMarkidis_double<4, 16, half, float, float, true>(A, B, C, M, K, N, merges, 1.0);
}

template<>
flop_counts matmul_simpleMarkidis_double<2>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    std::pair<int, int> merges[16];
    for(int i = 0; i < 16; i++)
        merges[i] = {i/4, i%4};
    return matmul_simpleMarkidis_double<4, 16, half, float, double, false>(A, B, C, M, K, N, merges, 1.0);
}

template<>
flop_counts matmul_simpleMarkidis_double<3>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    std::pair<int, int> merges[16];
    for(int i = 0; i < 16; i++)
        merges[i] = {i/4, i%4};
    return matmul_simpleMarkidis_double<4, 16, half, float, double, false>(A, B, C, M, K, N, merges, 1 << 11);
}

template<>
flop_counts matmul_simpleMarkidis_double<4>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    std::pair<int, int> merges[25];
    for(int i = 0; i < 25; i++)
        merges[i] = {i/5, i%5};
    return matmul_simpleMarkidis_double<5, 25, half, float, double, false>(A, B, C, M, K, N, merges, 1 << 11);
}

template<>
flop_counts matmul_simpleMarkidis_double<5>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    std::pair<int, int> merges[] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    return matmul_simpleMarkidis_double<2, 4, float, float, float, false>(A, B, C, M, K, N, merges, 1.0);
}

template<>
flop_counts matmul_simpleMarkidis_double<6>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    std::pair<int, int> merges[] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    return matmul_simpleMarkidis_double<2, 4, float, float, double, false>(A, B, C, M, K, N, merges, 1.0);
}

template<>
flop_counts matmul_simpleMarkidis_double<7>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    std::pair<int, int> merges[] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    return matmul_simpleMarkidis_double<2, 4, float, float, double, false>(A, B, C, M, K, N, merges, 1 << 24);
}

template<>
flop_counts matmul_simpleMarkidis_double<8>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    std::pair<int, int> merges[] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    return matmul_simpleMarkidis_double<2, 4, float, double, double, false>(A, B, C, M, K, N, merges, 1.0);
}






#if SM_VERSION >= 800
template<int splitCount, int mergeCount>
flop_counts matmul_simpleMarkidis_double_double(double *A, double *B, double *C, size_t M, size_t K, size_t N,
                                                std::pair<int, int> mergePattern[mergeCount]) 
{
    assert((M % 16) == 0);
    assert((K % 16) == 0);
    assert((N % 16) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate cpu");

    size_t ASizeD = M * K * sizeof(double);
    size_t BSizeD = K * N * sizeof(double);
    size_t CSizeD = M * N * sizeof(double);
    
    PROFILE_SEGMENTS_SWITCH("allocate gpu");

    double *deviceA, *deviceB;
    double *deviceC;
    double *deviceCMerged;
    double *deviceAFull, *deviceBFull;
    cudaGetLastError();
    PRINT_ON_ERROR(cudaMalloc(&deviceA, ASizeD * splitCount));
    PRINT_ON_ERROR(cudaMalloc(&deviceB, BSizeD * splitCount));
    PRINT_ON_ERROR(cudaMalloc(&deviceC, CSizeD * mergeCount));
    PRINT_ON_ERROR(cudaMalloc(&deviceCMerged, CSizeD));
    PRINT_ON_ERROR(cudaMalloc(&deviceAFull, ASizeD));
    PRINT_ON_ERROR(cudaMalloc(&deviceBFull, BSizeD));

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");

    PRINT_ON_ERROR(cudaMemcpy(deviceAFull, A, ASizeD, cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(deviceBFull, B, BSizeD, cudaMemcpyHostToDevice));

    PROFILE_SEGMENTS_SWITCH("split");

    split_cuda_double_double<splitCount><<<DivRoundUp(M*K, 256), 256>>>(deviceAFull, deviceA, M * K);
    PRINT_ON_ERROR(cudaGetLastError());
    split_cuda_double_double<splitCount><<<DivRoundUp(K*N, 256), 256>>>(deviceBFull, deviceB, K * N);
    PRINT_ON_ERROR(cudaGetLastError());

    CUDA_DEVICE_SYNCHRONIZE();

    PROFILE_SEGMENTS_SWITCH("matmul");
    for(int i = 0; i < mergeCount; i++)
    {
        size_t aIndex = mergePattern[i].first * M * K;
        size_t bIndex = mergePattern[i].second * K * N;
        size_t cIndex = i * M * N;
        matmulTensorCores<double, double, 2>(deviceA + aIndex, deviceB + bIndex, deviceC + cIndex, M, K, N);
        double scale = std::pow(2048, mergePattern[i].first) * std::pow(2048, mergePattern[i].second);
        divide_cuda<double><<<DivRoundUp(M*N, 256), 256>>>(deviceC + cIndex, M*N, scale);
    }
    CUDA_DEVICE_SYNCHRONIZE();

    PROFILE_SEGMENTS_SWITCH("merge");
    merge_cuda_double<mergeCount><<<DivRoundUp(M*N, 256), 256>>>(deviceC, deviceCMerged, M*N);
    PRINT_ON_ERROR(cudaGetLastError());
    CUDA_DEVICE_SYNCHRONIZE();

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
flop_counts matmul_simpleMarkidis_double_double<0>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    std::pair<int, int> merges[] = {{2, 2}, {2, 1}, {1, 2}, {0, 2}, {1, 1}, {2, 0}, {0, 1}, {1, 0}, {0, 0}};
    return matmul_simpleMarkidis_double_double<3, 9>(A, B, C, M, K, N, merges);
}

template<>
flop_counts matmul_simpleMarkidis_double_double<1>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    //std::pair<int, int> merges[16];
    //for(int i = 0; i < 16; i++)
        //merges[i] = {i/4, i%4};
    std::pair<int, int> merges[] = {{3, 3}, {3, 2}, {2, 3}, {2, 2}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {3, 0}, {0, 3}, {2, 0}, {0, 2},  {1, 1}, {0, 1}, {1, 0}, {0, 0}};
    return matmul_simpleMarkidis_double_double<4, 16>(A, B, C, M, K, N, merges);
}
#endif
