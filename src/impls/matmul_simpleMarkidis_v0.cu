#include <assert.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "../matmul.h"
#include "../profiler.h"

__global__ void matmul(half *A, half *B, float *C, int M, int K, int N) 
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

void matmul_simpleMarkidis_v0(float *A, float *B, float *C, int M, int K, int N) 
{
    assert((M % 16) == 0);
    assert((K % 16) == 0);
    assert((N % 16) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate cpu");

    half *A0 = (half*)malloc(M * K * sizeof(half));
    half *A1 = (half*)malloc(M * K * sizeof(half));
    half *B0 = (half*)malloc(K * N * sizeof(half));
    half *B1 = (half*)malloc(K * N * sizeof(half));
    float *hostC[4];
    for(int i = 0; i < 4; i++)
        hostC[i] = (float*)malloc(M * N * sizeof(float));

    PROFILE_SEGMENTS_SWITCH("split");

    split(A, A0, A1, M, K);
    split(B, B0, B1, K, N);

    PROFILE_SEGMENTS_SWITCH("allocate gpu");

    size_t ASize = M * K * sizeof(half);
    size_t BSize = K * N * sizeof(half);
    size_t CSize = M * N * sizeof(float);

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
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(M/threadsPerBlock.x, N/threadsPerBlock.y);
    matmul<<<blocks, threadsPerBlock>>>(deviceA[0], deviceB[0], deviceC[0], M, K, N);
    matmul<<<blocks, threadsPerBlock>>>(deviceA[0], deviceB[1], deviceC[1], M, K, N);
    matmul<<<blocks, threadsPerBlock>>>(deviceA[1], deviceB[0], deviceC[2], M, K, N);
    matmul<<<blocks, threadsPerBlock>>>(deviceA[1], deviceB[1], deviceC[3], M, K, N);

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

