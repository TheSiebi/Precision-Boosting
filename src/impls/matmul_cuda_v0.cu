#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>


__global__ void matmul_cuda_v0_kernel(double *A, double *B, double *C, int M, int K, int N) 
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double c_ij = 0.0;
            for (int k = 0; k < K; k++) {
                c_ij += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = c_ij;
        }
    }
}

extern "C"
{
#include "../matmul.h"

void matmul_cuda_v0(double *A, double *B, double *C, int M, int K, int N) 
{
    size_t ASize = M * K * sizeof(double);
    size_t BSize = K * N * sizeof(double);
    size_t CSize = M * N * sizeof(double);

    double *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, ASize);
    cudaMalloc(&deviceB, BSize);
    cudaMalloc(&deviceC, CSize);

    cudaMemcpy(deviceA, A, ASize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, BSize, cudaMemcpyHostToDevice);

    matmul_cuda_v0_kernel<<<1, 1>>>(deviceA, deviceB, deviceC, M, K, N);

    cudaMemcpy(C, deviceC, CSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}
}
