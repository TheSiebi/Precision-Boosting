#include <cuda_fp16.h>

#include "../split.h"

void split_v0(const double *A, void *A16, void *dA16, int M, int N)
{
    half *_A16 = (half *) A16;
    half *_dA16 = (half *) dA16;
    for (int i = 0; i < M * N; i++) {
        _A16[i] = __double2half(A[i]);
        double reconstructed = (double) __half2float(_A16[i]);
        _dA16[i] = __double2half(A[i] - reconstructed);
        
        // printf("A[%d] = %f\n", i, A[i]);
        // printf("A16[%d] = %f\n", i, __half2float(_A16[i]));
        // printf("dA16[%d] = %f\n", i, __half2float(_dA16[i]));
        // printf("A16[%d] + dA16[%d] - A[%d] = %f\n", i, i, i, ((double) __half2float(_A16[i]) + (double) __half2float(_dA16[i])) - A[i] );
    }    
}

void splitf_v0(const float *A, void *A16, void *dA16, int M, int N)
{
    half *_A16 = (half *) A16;
    half *_dA16 = (half *) dA16;
    for (int i = 0; i < M * N; i++) {
        _A16[i] = __float2half(A[i]);
        float reconstructed = __half2float(_A16[i]);
        _dA16[i] = __float2half(A[i] - reconstructed);
    }    
}

void split_Ootomo_v0(const double *A, void *A16, void *dA16, int M, int N)
{
    half *_A16 = (half *) A16;
    half *_dA16 = (half *) dA16;
    for (int i = 0; i < M * N; i++) {
        _A16[i] = __double2half(A[i]);
        double reconstructed = (double) __half2float(_A16[i]);
        _dA16[i] = __double2half((A[i] - reconstructed) * 2048.0);
    }
}

void splitf_Ootomo_v0(const float *A, void *A16, void *dA16, int M, int N)
{
    half *_A16 = (half *) A16;
    half *_dA16 = (half *) dA16;
    for (int i = 0; i < M * N; i++) {
        _A16[i] = __float2half(A[i]);
        float reconstructed = __half2float(_A16[i]);
        _dA16[i] = __float2half((A[i] - reconstructed) * 2048.0);
    }
}

