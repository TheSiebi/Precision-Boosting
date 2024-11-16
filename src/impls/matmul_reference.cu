#include "../timer.h"
#include "../matmul.h"

flop_counts matmul_reference32(float *A, float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float c_ij = 0.0;
            for (int k = 0; k < K; k++) {
                c_ij += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = c_ij;
        }
    }

    return {0L, 2L*M*K*N, 0L};
}

flop_counts matmul_reference64(double *A, double *B, double *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double c_ij = 0.0;
            for (int k = 0; k < K; k++) {
                c_ij += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = c_ij;
        }
    }

    return {0L, 0L, 2L*M*K*N};
}
