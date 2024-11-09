#include <stdio.h>
#include <stdlib.h>
#include "matmul.h"
#include "timer.h"
#include "profiler.h"
#include "math.h"
#include "split.h"
#include "merge_accumulate.h"
#include "rand.h"

#include <iostream>
#include <iomanip>
#include <string.h>

#define ARRAY_COUNT(arr) (sizeof(arr) / sizeof(arr[0]))

matmul_variant<float> matmulVariants32[] =
{
    {
        .function = matmul_simpleMarkidis_v0,
        .name = "Simple Markidis v0",
        .description = "Simple markidis with simple cuda matmul",
        .countFlops = matmul_flopcount_32,
    },
    {
        .function = matmul_simpleMarkidis_v1,
        .name = "Simple Markidis v1",
        .description = "Simple markidis with simple tensor matmul",
        .countFlops = matmul_flopcount_32,
    },
    {
        .function = matmul_simpleOotomo_v0,
        .name = "Simple Ootomo v0",
        .description = "Very basic Ootomo using CUDA",
    },
    {
        .function = matmul_Oootomo_v0,
        .name = "Ootomo v0",
        .description = "Ootomo with separate split, merge and matmul kernels (no accumulation outside tensor cores)",
        .countFlops = matmul_flopcount_32
    },
    {
        .function = matmul_Oootomo_v1,
        .name = "Ootomo v1",
        .description = "Ootomo algorithm as described by Code3 in the paper",
        .countFlops = matmul_flopcount_32
    },
    {
        .function = matmul_cuBLAS32,
        .name = "matmul_cuBLAS",
        .description = "cuBLAS",
        .countFlops = matmul_flopcount_32,
    }
};

matmul_variant<double> matmulVariants64[] =
{
    {
        .function = matmul_cuda_v0,
        .name = "matmul_cuda_v0",
        .description = "straightforward triple for loop implementation running on the GPU",
        .countFlops = matmul_flopcount_64,
    },
    {
        .function = matmul_cuBLAS64,
        .name = "matmul_cuBLAS",
        .description = "cuBLAS",
        .countFlops = matmul_flopcount_64,
    }
};

struct split_variant splitVariants[] =
{
    {
        .function = split_v0,
        .functionf = splitf_v0,
        .invFunction = merge_v0,
        .invFunctionf = mergef_v0,
        .name = "split_v0",
        .description = "straightforward cpu implementation of Markidis two way split",
    },
    {
        .function = split_Ootomo_v0,
        .functionf = splitf_Ootomo_v0,
        .invFunction = merge_Ootomo_v0,
        .invFunctionf = mergef_Ootomo_v0,
        .name = "split_Ootomo_v0",
        .description = "straightforward cpu implementation of Ootomo two way split",
    }
};

template<class T>
void referenceMatmul(T *A, T *B, T *C, int M, int K, int N) 
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            T c_ij = 0.0;
            for (int k = 0; k < K; k++) {
                c_ij += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = c_ij;
        }
    }
}

bool isEqual(const double *A, const double *B, int N)
{
    double epsilon = 0.001;
    _Bool fail = false;
    for (int i = 0; i < N; i++) {
        double err = fabs(A[i] - B[i]);
        if (err > epsilon) {
            fail = true;
            break;
        }
    }
    return !fail;
}

bool isEqual(const float *A, const float *B, int N)
{
    float epsilon = 0.002;
    _Bool fail = false;
    for (int i = 0; i < N; i++) {
        float err = fabsf(A[i] - B[i]);
        if (err > epsilon) {
            fail = true;
            break;
        }
    }
    return !fail;
}

void testSplitCorrectness(struct split_variant *function)
{
    int M, N;
    M = N = 32;
    double* A = (double *) malloc(M * N * sizeof(double));
    float* Af = (float *) malloc(M * N * sizeof(float));
    double* A_merged = (double *) malloc(M * N * sizeof(double));
    float* Af_merged = (float *) malloc(M * N * sizeof(float));
    void *A16 = malloc(M * N * 2);
    void *dA16 = malloc(M * N * 2);

    LCG rng = new_rng();
    
    // Populate matrices with random values between 0 and 1
    gen_urand<double>(&rng, A, M*N);
    
    // f_inv(f(x)) ~= identity
    function->function(A, A16, dA16, M, N);
    function->invFunction(A16, dA16, A_merged, M, N);
    _Bool fail = false;
    if (!isEqual(A, A_merged, M*N)) {
        printf("FAILURE: merging the output of %s (double variant) is not identical to input!\n", function->name);
        fail = true;
    } 

    // Populate matrices with random values between 0 and 1
    gen_urand<float>(&rng, Af, M*N);
    
    // f_inv(f(x)) ~= identity
    function->functionf(Af, A16, dA16, M, N);
    function->invFunctionf(A16, dA16, Af_merged, M, N);
    if (!isEqual(Af, Af_merged, M*N)) {
        printf("FAILURE: merging the output of %s (float variant) is not identical to input!\n", function->name);
        fail = true;
    } 

    if (!fail) {
        printf("SUCCESS: %s passed correctness tests!\n", function->name);
    }

    free(A);
    free(Af);
    free(A_merged);
    free(Af_merged);
    free(A16);
    free(dA16);
}

template<class T>
void printMatrix(T *A, int M, int N)
{
    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < N; n++)
        {
            if(n != 0)
                printf(", ");
            printf("%f", A[m*N + n]);
        }
        printf("\n");
    }
}

template<class T>
void testMatmulCorrectness_show_error(matmul_variant<T>* function, LCG rng)
{
    const T EPSILON = 0.002;
    const int FUNCTION_NAME_WIDTH = 26;
    const auto count_digits = [](int num)
    {
        if (num < 0)
            return 0; // unimplemented
        if (num == 0)
            return 1;
        int ret = 0;
        while (num > 0)
        {
            num /= 10;
            ret++;
        }
        return ret;
    };

    // Parameters
    // srand(time(NULL));
    const size_t M = 1024, K = 512, N = 256;
    // printf("Settings: M = %lu, K = %lu, N = %lu\n", M, K, N);

    // Allocate matrices
    T *A = (T*) malloc(M * K * sizeof(T));
    T *B = (T*) malloc(K * N * sizeof(T));
    T *C = (T*) calloc(M * N, sizeof(T)); // Ensures C is zero to avoid interference from previous runs.
    T *C_ref = (T*) malloc(M * N * sizeof(T));

    // Populate A, B, C
    gen_urand<T>(&rng, A, M * K);
    gen_urand<T>(&rng, B, K * N);

    // Compute reference solution
    referenceMatmul(A, B, C_ref, M, K, N);

    // Call function under test
    function->function(A, B, C, M, K, N);

    // Analyze result
    bool success = true;
    int wrong = 0; // index of first wrong matrix entry
    T abs_diff = 0.f, rel_errr = 0.f;
    for (size_t i = 0; i < M * N; ++i)
    {
        abs_diff = std::max(abs_diff, abs(C_ref[i] - C[i]));
        rel_errr = std::max(rel_errr, abs(C_ref[i] - C[i]) / C_ref[i]);
        if (abs_diff > EPSILON)
        {
            std::cout
                << "\033[31m" << "[ERROR]    " << "\033[0m" // Red text
                << function->name;
            for (int u = 0; u < FUNCTION_NAME_WIDTH - (int) strlen(function->name); ++u)
                std::cout << " ";
            std::cout
                << "wrong @ " << i << " ";
            int spaces = 2 + 3 - count_digits(i);
            for (int tmp = 0; tmp < spaces; ++tmp)
                std::cout << " ";
            success = false;
            wrong = i;
            break;
        }
    }
    if (success)
    {
        std::cout
            << "\033[32m" << "[SUCCESS]  " << "\033[0m" // Green text
            << function->name;
        for (int u = 0; u < FUNCTION_NAME_WIDTH - (int) strlen(function->name); ++u)
            std::cout << " ";
        std::cout << "correct.      ";
    }
    std::cout
        << "abs \033[33m" << std::setw(12) << abs_diff << "\033[0m  " // Yellow text
        << "rel \033[33m" << std::setw(12) << rel_errr  * 100.f << " %\033[0m"; // Yellow text
    if (!success)
    {
        std::cout
            << "  E "
            << C_ref[wrong]
            << " != "
            << C[wrong]
            << " A ";
    }
    std::cout << std::endl;

    // Free memory
    free(A);
    free(B);
    free(C);
    free(C_ref);
}

template<class T>
void testMatmulCorrectness(matmul_variant<T> *function, LCG rng) {
    // A * B = C
    // A is m*k (m rows, k columns)
    // B is k*n (k rows, n columns)
    // C is m*n (m rows, n columns)
    int M, K, N;
    M = K = N = 32;
    T* A = (T *) malloc(M * K * sizeof(T));
    T* B = (T *) malloc(K * N * sizeof(T));
    T* C = (T *) malloc(M * N * sizeof(T));
    T* C_ref = (T *) malloc(M * N * sizeof(T));

    // Populate matrices with random values between 0 and 1
    gen_urand<T>(&rng, A, M*K);
    gen_urand<T>(&rng, B, K*N);

    // Use matmul_v0 as a reference implementation
    referenceMatmul(A, B, C_ref, M, K, N);
    function->function(A, B, C, M, K, N);
    
    if (!isEqual(C, C_ref, M*N)) 
    {
        printf("FAILURE: %s is NOT identical to the reference\n", function->name);
        printf("Result\n");
        printMatrix(C, M, N);
        printf("Expected\n");
        printMatrix(C_ref, M, N);
    } else {
        printf("SUCCESS: %s passed correctness tests!\n", function->name);
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

template<class T>
void profile(matmul_variant<T> variant, int warmup, int iterations, int M, int K, int N)
{
    T *A = (T*)calloc(M * K, sizeof(*A));
    T *B = (T*)calloc(K * N, sizeof(*B));
    T *C = (T*)calloc(M * N, sizeof(*C));

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
        LCG rng = new_rng();
        for(size_t i = 0; i < ARRAY_COUNT(matmulVariants32); i++)
        {   
            testMatmulCorrectness_show_error(&matmulVariants32[i], rng);
            // testMatmulCorrectness(&matmulVariants32[i]);
        }
        for(size_t i = 0; i < ARRAY_COUNT(matmulVariants64); i++)
        {
            testMatmulCorrectness_show_error(&matmulVariants64[i], rng);
            // testMatmulCorrectness(&matmulVariants64[i]);
        }
        for(size_t i = 0; i < ARRAY_COUNT(splitVariants); i++)
        {
            testSplitCorrectness(&splitVariants[i]);
        }
        // profile(matmulVariants64[0], 0, 1, 4096, 4096, 4096);
        // profile(matmulVariants64[1], 0, 1, 4096, 4096, 4096);
        // profile(matmulVariants32[0], 0, 1, 8192, 8192, 8192);
        // profile(matmulVariants32[1], 0, 1, 8192, 8192, 8192);
        // profile(matmulVariants32[3], 0, 1, 8192, 8192, 8192);
        // TODO: profiling cuBLAS32 gives me a huge time, not sure why
        //profile(matmulVariants32[4], 0, 1, 8192, 8192, 8192);

    }
    else
    {
        for(size_t i = 0; i < ARRAY_COUNT(matmulVariants64); i++)
        {
            timeFunction(&matmulVariants64[i], argv[2]);
        }
    }
    return 0;
}
