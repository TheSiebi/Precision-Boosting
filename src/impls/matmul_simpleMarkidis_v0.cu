#include <assert.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../matmul.h"
#include "../profiler.h"

__global__ void matmul_v0(half *A, half *B, float *C, int M, int K, int N) 
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    float result = 0.0;
    for (int k = 0; k < K; k++) 
    {
        result += (float)(A[m*K + k] * B[k*N + n]);
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

void split(const float *A, void *A16, void *dA16, int M, int N)
{
    half *_A16 = (half *) A16;
    half *_dA16 = (half *) dA16;
    for (int i = 0; i < M * N; i++) {
        _A16[i] = __float2half(A[i]);
        float reconstructed = __half2float(_A16[i]);
        _dA16[i] = __float2half(A[i] - reconstructed);
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

    half *A0 = (half*)malloc(ASize);
    half *A1 = (half*)malloc(ASize);
    half *B0 = (half*)malloc(BSize);
    half *B1 = (half*)malloc(BSize);
    float *hostC[4];
    for(int i = 0; i < 4; i++)
        hostC[i] = (float*)malloc(CSize);

    PROFILE_SEGMENTS_SWITCH("split");

    split(A, A0, A1, M, K);
    split(B, B0, B1, K, N);

    PROFILE_SEGMENTS_SWITCH("allocate gpu");


    half *deviceA[2], *deviceB[2];
    float *deviceC[4];
    for(int i = 0; i < 2; i++)
    {
        cudaMalloc(&deviceA[i], ASize);
        cudaMalloc(&deviceB[i], BSize);
    }
    for(int i = 0; i < 4; i++)
        cudaMalloc(&deviceC[i], CSize);

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    cudaMemcpy(deviceA[0], A0, ASize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceA[1], A1, ASize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB[0], B0, BSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB[1], B1, BSize, cudaMemcpyHostToDevice);

    PROFILE_SEGMENTS_SWITCH("matmul");
    if constexpr (version == 0)
    {
        dim3 threadsPerBlock(16, 16);
        dim3 blocks(M/threadsPerBlock.x, N/threadsPerBlock.y);
        for(int i = 0; i < 4; i++)
            matmul_v0<<<blocks, threadsPerBlock>>>(deviceA[i/2], deviceB[i%2], deviceC[i], M, K, N);
    }
    else if constexpr (version == 1)
    {
        dim3 threadsPerBlock(32, 1);
        dim3 blocks(M/16, N/16);
        for(int i = 0; i < 4; i++)
            matmul_v1<<<blocks, threadsPerBlock>>>(deviceA[i/2], deviceB[i%2], deviceC[i], M, K, N);
    }

    cudaDeviceSynchronize();

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    for(int i = 0; i < 4; i++)
        cudaMemcpy(hostC[i], deviceC[i], CSize, cudaMemcpyDeviceToHost);

    PROFILE_SEGMENTS_SWITCH("merge");
    for(int i = 0; i < M * N; i++)
        C[i] = hostC[0][i] + hostC[1][i] + hostC[2][i] + hostC[3][i];

    PROFILE_SEGMENTS_SWITCH("free");
    free(A0);
    free(A1);
    free(B0);
    free(B1);
    for(int i = 0; i < 4; i++)
        free(hostC[i]);

    for(int i = 0; i < 2; i++)
    {
        cudaFree(&deviceA[i]);
        cudaFree(&deviceB[i]);
    }
    for(int i = 0; i < 4; i++)
        cudaFree(&deviceC[i]);
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

