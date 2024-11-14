#include <cuda_fp16.h>

#include "../split.h"

// TOTAL: M*N flops64
void split_v0(const double *A, void *A16, void *dA16, int M, int N)
{
    half *_A16 = (half *) A16;
    half *_dA16 = (half *) dA16;
    // TOTAL: M*N flops64
    for (int i = 0; i < M * N; i++) {
        _A16[i] = __double2half(A[i]);
        double reconstructed = (double) __half2float(_A16[i]);
        // 1 flop64
        _dA16[i] = __double2half(A[i] - reconstructed);
    }    
}

// TOTAL: M*N flops32
void splitf_v0(const float *A, void *A16, void *dA16, int M, int N)
{
    half *_A16 = (half *) A16;
    half *_dA16 = (half *) dA16;
    // TOTAL: M*N flops32
    for (int i = 0; i < M * N; i++) {
        _A16[i] = __float2half(A[i]);
        float reconstructed = __half2float(_A16[i]);
        // 1 flop32
        _dA16[i] = __float2half(A[i] - reconstructed);
    }    
}

// TOTAL: 2*M*N flops64
void split_Ootomo_v0(const double *A, void *A16, void *dA16, int M, int N)
{
    half *_A16 = (half *) A16;
    half *_dA16 = (half *) dA16;
    // TOTAL: 2*M*N flops64
    for (int i = 0; i < M * N; i++) {
        _A16[i] = __double2half(A[i]);
        double reconstructed = (double) __half2float(_A16[i]);
        // 2 flops64
        _dA16[i] = __double2half((A[i] - reconstructed) * 2048.0);
    }
}

// TOTAL: 2*M*N flops32
void splitf_Ootomo_v0(const float *A, void *A16, void *dA16, int M, int N)
{
    half *_A16 = (half *) A16;
    half *_dA16 = (half *) dA16;
    // TOTAL: 2*M*N flops32
    for (int i = 0; i < M * N; i++) {
        _A16[i] = __float2half(A[i]);
        float reconstructed = __half2float(_A16[i]);
        // 2 flops32
        _dA16[i] = __float2half((A[i] - reconstructed) * 2048.0);
    }
}

