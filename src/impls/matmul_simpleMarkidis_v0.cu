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

__global__ void matmul_v0(half *A, half *B, float *C, int M, int K, int N) 
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    float result = 0.0;
    for (int k = 0; k < K; k++) 
    {
        result += (float)A[m*K + k] * (float)B[k*N + n];
    }
    C[m*N + n] = result;
}

__global__ void matmul_v1(half *A, half *B, float *C, int M, int K, int N) 
{
    using namespace nvcuda;

    int warpM = blockIdx.x * 16;
    int warpN = blockIdx.y * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;

    wmma::fill_fragment(cFrag, 0);

    for (int k = 0; k < K; k += 16) 
    {
        wmma::load_matrix_sync(aFrag, A + warpM * K + k, K);
        wmma::load_matrix_sync(bFrag, B + k * N + warpN, N);

        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    wmma::store_matrix_sync(C + warpM * N + warpN, cFrag, N, wmma::mem_row_major);
}

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
    if constexpr (version == 0)
    {
        dim3 threadsPerBlock(16, 16);
        dim3 blocks(M/threadsPerBlock.x, N/threadsPerBlock.y);
        for(int i = 0; i < 4; i++)
        {
            matmul_v0<<<blocks, threadsPerBlock>>>(deviceA[i/2], deviceB[i%2], deviceC[i], M, K, N);
            PRINT_ON_ERROR(cudaGetLastError());
        }
    }
    else if constexpr (version == 1)
    {
        dim3 threadsPerBlock(32, 1);
        dim3 blocks(M/16, N/16);
        for(int i = 0; i < 4; i++)
        {
            matmul_v1<<<blocks, threadsPerBlock>>>(deviceA[i/2], deviceB[i%2], deviceC[i], M, K, N);
            PRINT_ON_ERROR(cudaGetLastError());
        }
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

void matmul_simpleMarkidis_v0(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_simpleMarkidis<0>(A, B, C, M, K, N);
}

void matmul_simpleMarkidis_v1(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_simpleMarkidis<1>(A, B, C, M, K, N);
}

