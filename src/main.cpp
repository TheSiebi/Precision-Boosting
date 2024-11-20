#include <stdio.h>
#include <stdlib.h>
#include "matmul.h"
#include "timer.h"
#include "profiler.h"
#include "math.h"
#include "split.h"
#include "merge_accumulate.h"
#include "rand.h"
#include "precision.h"
#include "matcache.h"

#include <iostream>
#include <iomanip>
#include <string.h>

#define ARRAY_COUNT(arr) (sizeof(arr) / sizeof(arr[0]))

const int FUNCTION_NAME_WIDTH = 40;

matmul_variant<float> matmulVariants32[] =
{
    {
        .function = matmul_simpleMarkidis<0>,
        .name = "Simple Markidis v0",
        .description = "Simple markidis with simple cuda matmul",
    },
    {
        .function = matmul_simpleMarkidis<1>,
        .name = "Simple Markidis v1",
        .description = "Simple markidis with simple tensor matmul",
    },
    {
        .function = matmul_simpleMarkidis<2>,
        .name = "Simple Markidis v2",
        .description = "Simple markidis with multiple warps per block",
    },
    {
        .function = matmul_simpleMarkidis<3>,
        .name = "Simple Markidis v3",
        .description = "Simple markidis with shared memory",
    },
    {
        .function = matmul_simpleMarkidis<4>,
        .name = "Simple Markidis v4",
        .description = "Simple markidis with shared memory",
    },
    {
        .function = matmul_markidis,
        .name = "Markidis",
        .description = "Markidis in a single cuda kernel",
    },
    {
        .function = matmul_basic_Ootomo_v0,
        .name = "Basic Ootomo v0",
        .description = "Very basic Ootomo using CUDA",
    },
    {
        .function = matmul_Ootomo_v0,
        .name = "Ootomo v0",
        .description = "Ootomo with separate split, merge and matmul kernels (no accumulation outside tensor cores)",
    },
    {
        .function = matmul_Ootomo_v1,
        .name = "Ootomo v1",
        .description = "Ootomo algorithm as described by Code3 in the paper",
    },
    {
        .function = matmul_Ootomo_v2,
        .name = "Ootomo v2",
        .description = "Same as Ootomo_v1 but with better data reuse",
    },
    {
        .function = matmul_cuBLAS32,
        .name = "matmul_cuBLAS",
        .description = "cuBLAS",
    }
};

matmul_variant<double> matmulVariants64[] =
{
    {
        .function = matmul_cuda<double, double, 0>,
        .name = "matmul_cuda v0",
        .description = "straightforward triple for loop implementation running on the GPU",
    },
    {
        .function = matmul_cuda<double, double, 1>,
        .name = "matmul_cuda v1",
        .description = "straightforward triple for loop implementation running on the GPU",
    },
    {
        .function = matmul_cuda<double, double, 2>,
        .name = "matmul_cuda v2",
        .description = "straightforward triple for loop implementation running on the GPU",
    },
    {
        .function = matmul_cuda<double, double, 3>,
        .name = "matmul_cuda v3",
        .description = "straightforward triple for loop implementation running on the GPU",
    },
    {
        .function = matmul_cuBLAS64,
        .name = "matmul_cuBLAS",
        .description = "cuBLAS",
    },
    {
        .function = matmul_Ozaki_v0,
        .name = "matmul_Ozaki v0",
        .description = "Ozaki FP64 using FP32 on CPU",
    },
    {
        .function = matmul_Ootomo_double_v0,
        .name = "Ootomo double v0",
        .description = "Use external split to partition double into 4 float multiplications. Perform this 4 float multiplications with fp32 Ootomo. Merge the 4 results.",        
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

void testSplitCorrectness(struct split_variant *function, LCG* rng)
{
    int M, N;
    M = N = 32;
    double* A = (double *) malloc(M * N * sizeof(double));
    float* Af = (float *) malloc(M * N * sizeof(float));
    double* A_merged = (double *) malloc(M * N * sizeof(double));
    float* Af_merged = (float *) malloc(M * N * sizeof(float));
    void *A16 = malloc(M * N * 2);
    void *dA16 = malloc(M * N * 2);

    double absolute_residual_sum = 0.0;
    double relative_residual_sum = 0.0;

    // Populate matrices with random values between -1 and 1
    gen_urand<double>(rng, A, M*N);
    
    // f_inv(f(x)) ~= identity
    function->function(A, A16, dA16, M, N);
    function->invFunction(A16, dA16, A_merged, M, N);
    bool fail = false;
    if (!test_equality(A_merged, A, M*N)) {
        printf("FAILURE: merging the output of %s (double variant) is not identical to input!\n", function->name);
        fail = true;
    }

    absolute_residual_sum += abs_residual(A_merged, A, M * N);
    relative_residual_sum += rel_residual(A_merged, A, M * N);

    // Populate matrices with random values between -1 and 1
    gen_urand<float>(rng, Af, M*N);
    
    // f_inv(f(x)) ~= identity
    function->functionf(Af, A16, dA16, M, N);
    function->invFunctionf(A16, dA16, Af_merged, M, N);
    if (!test_equality(Af_merged, Af, M*N)) {
        printf("FAILURE: merging the output of %s (float variant) is not identical to input!\n", function->name);
        fail = true;
    }
    
    absolute_residual_sum += abs_residual(Af_merged, Af, M * N);
    relative_residual_sum += rel_residual(Af_merged, Af, M * N);

    double avg_rel_residual = relative_residual_sum / 2.0;
    double avg_abs_residual = absolute_residual_sum / 2.0;
        
    if (!fail) {
        // Success!
        std::cout
            << "\033[32m" << "[SUCCESS]  " << "\033[0m" // Green text
            << std::left << std::setw(FUNCTION_NAME_WIDTH) << function->name;
        // Print residual errors
        std::cout << "Avg residual: \033[33m" << std::left << std::setw(11) << avg_rel_residual << "\033[0m (rel) \033[33m" << std::left << std::setw(11) << avg_abs_residual << "\033[0m (abs)" << std::endl;
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
void testMatmulCorrectness(matmul_variant<T>* function, LCG *rng)
{
    // Parameters
    // srand(time(NULL));
    const size_t RUNS = 5;
    bool failed = false;
    double relative_residual_sum = 0.0;
    double absolute_residual_sum = 0.0;
    for (size_t run = 0; run < RUNS; run++) {
        uint64_t starting_seed = rng->state;
        // Randomize matrix dimensions
        size_t M, K, N;
        M = 1 << next_int(rng, 7, 9);
        K = 1 << next_int(rng, 7, 9);
        N = 1 << next_int(rng, 7, 9);

        // Get matrices
        T *C = (T*) calloc(M * N,  sizeof(T));
        auto [A, B, C_reference] = getMatrices<T>(M, K, N, "uniform", rng);

        // Call function under test
        function->function(A, B, C, M, K, N);

        // Analyze result
        int wrong = test_matmul_correctness_full(C, C_reference, M, N);
        double absolute_residual = abs_residual(C, C_reference, M * N);
        double relative_residual = rel_residual(C, C_reference, M * N);
        absolute_residual_sum += absolute_residual;
        relative_residual_sum += relative_residual;

        if (wrong != -1) {
            failed = true;
            // Give a nice error message
            double wrong_val = (double) C[wrong];
            double ref_sol = C_reference[wrong];
            double abs_err = abs(ref_sol - wrong_val);
            double rel_err = abs_err / abs(ref_sol);
            size_t row = wrong / N;
            size_t col = wrong % N;

            std::cout
                << "\033[31m" << "[FAILURE]  " // Red text
                << "\033[0m" << std::left << std::setw(FUNCTION_NAME_WIDTH) << function->name
                << "\t" << "Residual: \033[33m" << relative_residual << "\033[0m (rel) \033[33m" << absolute_residual << "\033[0m (abs)" << std::endl;
            
            std::cout << "\t" << "Seed: \033[33m" << std::hex << starting_seed << std::dec << "\033[0m\tM=\033[33m" << M << "\033[0m K=\033[33m" << K << "\033[0m N=\033[33m" << N << "\033[0m" << std::endl;
            std::cout << "\t" << "Wrong at: \033[33mRow " << row << "\033[0m, \033[33mCol " << col << "\033[0m" << std::endl;
            std::cout << "\t" << "Expected: \033[33m" << ref_sol << "\033[0m\tActual:   \033[33m" << wrong_val << "\033[0m" << std::endl;
            std::cout << "\t" << "Error:    \033[33m" << rel_err << "\033[0m (rel) \033[33m" << abs_err << "\033[0m (abs)" << std::endl;
            break;
        }

        // Free memory
        free(A);
        free(B);
        free(C);
    }

    if (!failed) {
        // Success!
        double avg_rel_residual = relative_residual_sum / (double) RUNS;
        double avg_abs_residual = absolute_residual_sum / (double) RUNS;
        std::cout
            << "\033[32m" << "[SUCCESS]  " << "\033[0m" // Green text
            << std::left << std::setw(FUNCTION_NAME_WIDTH) << function->name;
        // Print residual errors
        std::cout << "Avg residual: \033[33m" << std::left << std::setw(11) << avg_rel_residual << "\033[0m (rel) \033[33m" << std::left << std::setw(11) << avg_abs_residual << "\033[0m (abs)" << std::endl;
    }
}

template<class T>
void profile(matmul_variant<T> variant, int warmup, int iterations, int M, int K, int N)
{
    T *A = (T*)calloc(M * K, sizeof(*A));
    T *B = (T*)calloc(K * N, sizeof(*B));
    T *C = (T*)calloc(M * N, sizeof(*C));
    flop_counts counts;

    profiler_reset();
    for(int i = 0; i < warmup; i++)
        variant.function(A, B, C, M, K, N);
    profiler_reset();
    for(int i = 0; i < iterations; i++)
        counts = variant.function(A, B, C, M, K, N);
    printf("\n----Profiling %s------\n", variant.name);
    profiler_segments_print(counts.flops16, counts.flops32, counts.flops64);

    free(A);
    free(B);
    free(C);
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        //LCG rng = new_rng();
        LCG rng = rng_seeded(0xC0FEE);
        uint64_t seed = rng.state;
        printf("\nRunning tests with seed: %lx\n\n", rng.state);
        for(size_t i = 0; i < ARRAY_COUNT(matmulVariants32); i++)
        {   
            rng.state = seed;
            testMatmulCorrectness(&matmulVariants32[i], &rng);
        }
        for(size_t i = 0; i < ARRAY_COUNT(matmulVariants64); i++)
        {
            rng.state = seed;
            testMatmulCorrectness(&matmulVariants64[i], &rng);
        }
        for(size_t i = 0; i < ARRAY_COUNT(splitVariants); i++)
        {
            rng.state = seed;
            testSplitCorrectness(&splitVariants[i], &rng);
        }
        
        /*        
        profile(matmulVariants64[0], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants64[1], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants64[2], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants64[3], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants64[4], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants64[5], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants64[4], 0, 1, 4096, 4096, 4096);
        */

        
        profile(matmulVariants32[1], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants32[2], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants32[3], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants32[4], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants32[5], 0, 1, 8192, 8192, 8192);
        /*
        profile(matmulVariants32[6], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants32[7], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants32[8], 0, 1, 8192, 8192, 8192);
        
        profile(matmulVariants32[9], 0, 1, 8192, 8192, 8192);
        */        
    }
    else
    {
        LCG rng = new_rng();
        uint64_t seed = rng.state;

        std::vector<int> timeIndices64 = {};
        for(const int value : timeIndices64)
        {
            timeFunction(&matmulVariants64[value], argv[2], rng);
            rng.state = seed;
        }
        
        
        std::vector<int> timeIndices32 = {7, 8, 9};
        for(const int value : timeIndices32)
        {
            timeFunction(&matmulVariants32[value], argv[2], rng);
            rng.state = seed;
        }
    }
    return 0;
}
