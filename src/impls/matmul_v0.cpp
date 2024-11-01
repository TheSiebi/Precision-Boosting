#include "../profiler.h"

void matmul_v0(double *A, double *B, double *C, int M, int K, int N) 
{
    PROFILE_FUNCTION_START();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double c_ij = 0.0;
            for (int k = 0; k < K; k++) {
                c_ij += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = c_ij;
        }
    }
    PROFILE_FUNCTION_END();
}
