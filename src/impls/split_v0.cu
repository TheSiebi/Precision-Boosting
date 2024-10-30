#include <cuda_fp16.h>

extern "C"
{
#include "../split.h"

void split_v0(const double *A, void *A16, void *dA16, int M, int N)
{
    half *_A16 = (half *) A16;
    half *_dA16 = (half *) dA16;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            _A16[i*N + j] = __double2half(A[i*N + j]);
            double reconstructed = (double) __half2float(_A16[i*N + j]);
            _dA16[i*N + j] = __double2half(A[i*N + j] - reconstructed);
            
            // printf("A[%d][%d] = %f\n", i, j, A[i*N + j]);
            // printf("A16[%d][%d] = %f\n", i, j, __half2float(_A16[i*N + j]));
            // printf("dA16[%d][%d] = %f\n", i, j, __half2float(_dA16[i*N + j]));
            // printf("A16[%d][%d] + dA16[%d][%d] - A[%d][%d] = %f\n", i, j, i, j, i, j, ((double) __half2float(_A16[i*N + j]) + (double) __half2float(_dA16[i*N + j])) - A[i*N + j] );
        }
    }    
}

void split_Ootomo_v0(const double *A, void *A16, void *dA16, int M, int N)
{
    half *_A16 = (half *) A16;
    half *_dA16 = (half *) dA16;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            _A16[i*N + j] = __double2half(A[i*N + j]);
            double reconstructed = (double) __half2float(_A16[i*N + j]);
            _dA16[i*N + j] = __double2half((A[i*N + j] - reconstructed) * 2048.0);
        }
    }
}
}

