#include <stdio.h>
#include <stdlib.h>
#include "matmul.h"
#include "timer.h"
#include "profiler.h"
#include "math.h"
#include "split.h"

#define ARRAY_COUNT(arr) (sizeof(arr) / sizeof(arr[0]))

struct matmul_variant variants[] =
{
    {
        .function = matmul_v0,
        .name = "matmul_v0",
        .description = "straightforward triple for loop implementation",
    },
    {
        .function = matmul_cuda_v0,
        .name = "matmul_cuda_v0",
        .description = "straightforward triple for loop implementation running on the GPU",
    }
};

void testCorrectness(struct matmul_variant *function) {
    // A * B = C
    // A is m*k (m rows, k columns)
    // B is k*n (k rows, n columns)
    // C is m*n (m rows, n columns)
    int M, K, N;
    M = K = N = 32;
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

    // void *A16 = malloc(M * K * 2);
    // void *dA16 = malloc(M * K * 2);
    // split_v0(A, A16, dA16, M, K);

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

void profile(struct matmul_variant variant, int warmup, int iterations, int M, int K, int N)
{
    double *A = calloc(M * K, sizeof(*A));
    double *B = calloc(K * N, sizeof(*B));
    double *C = calloc(M * N, sizeof(*C));

    profiler_reset();
    for(int i = 0; i < warmup; i++)
        variant.function(A, B, C, M, K, N);
    profiler_reset();
    for(int i = 0; i < iterations; i++)
        variant.function(A, B, C, M, K, N);
    printf("\n----Profiling %s------\n", variant.name);
    profiler_segments_print();

    free(A);
    free(B);
    free(C);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        for(size_t i = 0; i < ARRAY_COUNT(variants); i++)
        {
            testCorrectness(&variants[i]);
        }
        profile(variants[0], 0, 1, 1024, 1024, 1024);
        profile(variants[1], 0, 1, 8192, 8192, 8192);
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
