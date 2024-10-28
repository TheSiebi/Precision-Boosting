#include <stdio.h>
#include <stdlib.h>
#include "matmul.h"
#include "timer.h"
#include "math.h"

#define ARRAY_COUNT(arr) (sizeof(arr) / sizeof(arr[0]))

struct matmul_variant variants[] =
{
    {
        .function = matmul_v0,
        .name = "matmul_v0",
        .description = "straightforward triple for loop implementation",
    }
};

void testCorrectness(struct matmul_variant *function) {
    // A * B = C
    // A is m*k (m rows, k columns)
    // B is k*n (k rows, n columns)
    // C is m*n (m rows, n columns)
    int M, K, N;
    M = K = N = 10;
    double* A = (double *) malloc(M * K * sizeof(double));
    double* B = (double *) malloc(K * N * sizeof(double));
    double* C = (double *) malloc(M * N * sizeof(double));
    double* C_ref = (double *) malloc(M * N * sizeof(double));

    // Populate matrices with random values between 0 and 1
    for (int j = 0; j < M*K; j++) {
        A[j] = (double) rand() / (double) RAND_MAX;
    }
    for (int j = 0; j < K*N; j++) {
        B[j] = (double) rand() / (double) RAND_MAX;
    }

    // Use matmul_v0 as a reference implementation
    matmul_v0(A, B, C_ref, M, K, N);
    function->function(A, B, C, M, K, N);

    double epsilon = 0.001;
    _Bool fail = false;
    for (int i = 0; i < M * N; i++) {
        double err = fabs(C[i] - C_ref[i]);
        if (err > epsilon) {
            fail = true;
            break;
        }
    }
    
    if (fail) {
        printf("FAILURE: %s is NOT identical to the reference\n", function->name);
    } else {
        printf("SUCCESS: %s passed correctness tests!\n", function->name);
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        for(size_t i = 0; i < ARRAY_COUNT(variants); i++)
        {
            testCorrectness(&variants[i]);
        }
    }
    else
    {
        for(size_t i = 0; i < ARRAY_COUNT(variants); i++)
        {
            timeFunction(&variants[i], argv[2]);
        }
    }
    return 0;
}
