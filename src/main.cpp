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
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <vector>

#define ARRAY_COUNT(arr) (sizeof(arr) / sizeof(arr[0]))

const int FUNCTION_NAME_WIDTH = 40;

matmul_variant<float> matmulVariants32[] =
{
    {
        .function = matmul_simpleMarkidis<0, 1, false>,
        .name = "Simple Markidis v0",
        .description = "Simple markidis with simple tensor matmul",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis<1, 1, false>,
        .name = "Simple Markidis v1",
        .description = "Simple markidis with multiple warps per block",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis<2, 1, false>,
        .name = "Simple Markidis v2",
        .description = "Simple markidis with shared memory",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis<3, 1, false>,
        .name = "Simple Markidis v3",
        .description = "Simple markidis with shared memory",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis<3, 4, false>,
        .name = "Simple Markidis v3 stream:4 scale:false",
        .description = "Simple markidis with shared memory",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis<4, 1, false>,
        .name = "Simple Markidis v4",
        .description = "Simple markidis with shared memory, vectorized loads",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis<4, 1, true>,
        .name = "Simple Markidis v4 scaled",
        .description = "Simple markidis with shared memory, vectorized loads and split scaling",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis<5, 1, false>,
        .name = "Simple Markidis v4 acc:Cuda",
        .description = "Simple markidis with shared memory, vectorized loads and accumulation outside tensor cores",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis<5, 1, true>,
        .name = "Simple Markidis v4 acc:Cuda, scaled",
        .description = "Simple markidis with shared memory, vectorized loads, accumulation outside tensor cores and split scaling",
        .highestPerforming = false,
    },
#if SM_VERSION >= 800
    {
        .function = matmul_simpleMarkidis<6, 1, false>,
        .name = "Simple Markidis v5 acc:Cuda",
        .description = "Simple markidis with shared memory, vectorized loads, double buffering, accumulation outside tensor cores and split scaling",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis<6, 1, true>,
        .name = "Simple Markidis v5 acc:Cuda, scaled",
        .description = "Simple markidis with shared memory, vectorized loads, double buffering, accumulation outside tensor cores and split scaling",
        .highestPerforming = true,
    },
    {
        .function = matmul_simpleMarkidis<7, 1, false>,
        .name = "Simple Markidis v5",
        .description = "Simple markidis with shared memory, vectorized loads, double buffering and split scaling",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis<7, 1, true>,
        .name = "Simple Markidis v5 scaled",
        .description = "Simple markidis with shared memory, vectorized loads, double buffering and split scaling",
        .highestPerforming = true,
    },
#endif
    {
        .function = matmul_markidis,
        .name = "Markidis",
        .description = "Markidis in a single cuda kernel",
        .highestPerforming = false,
    },
    {
        .function = matmul_basic_Ootomo_v0,
        .name = "Basic Ootomo v0",
        .description = "Very basic Ootomo using CUDA",
        .highestPerforming = false,
    },
    {
        .function = matmul_Ootomo_v0,
        .name = "Ootomo v0",
        .description = "Ootomo with separate split, merge and matmul kernels (no accumulation outside tensor cores)",
        .highestPerforming = false,
    },
    {
        .function = matmul_Ootomo_v1,
        .name = "Ootomo v1",
        .description = "Ootomo algorithm as described by Code3 in the paper",
        .highestPerforming = false,
    },
    {
        .function = matmul_Ootomo_v2,
        .name = "Ootomo v2",
        .description = "Same as Ootomo_v1 but with better data reuse. Accumulation inside tensor cores",
        .highestPerforming = true,
    },
    {
        .function = matmul_Ootomo_v3,
        .name = "Ootomo v3",
        .description = "Same as Ootomo_v2 but with accumulation outside tensor cores",
        .highestPerforming = true,
    },
    {
        .function = matmul_simpleMarkidis_float<0>,
        .name = "Hybrid float: 1 term",
        .description = "Split 2 half, 1 float multiply & accumulate (TENSOR CORES)",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis_float<1>,
        .name = "Hybrid float: 2 terms",
        .description = "Split 2 half, 2 float multiply & accumulate (TENSOR CORES)",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis_float<2>,
        .name = "Hybrid float: 2 terms",
        .description = "Split 2 half, 2 float multiply & accumulate (TENSOR CORES)",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis_float<3>,
        .name = "Hybrid float: 3 terms",
        .description = "Split 2 half, 3 float multiply & accumulate (TENSOR CORES)",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis_float<4>,
        .name = "Hybrid float: 4 terms",
        .description = "Split 2 half, 4 float multiply & accumulate (TENSOR CORES)",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis_float<5>,
        .name = "Hybrid float: 9 terms",
        .description = "Split 3 half, 9 float multiply & accumulate (TENSOR CORES)",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis_float<6>,
        .name = "Hybrid float: 16 terms",
        .description = "Split 4 half, 16 float multiply & accumulate (TENSOR CORES)",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis_float<7>,
        .name = "Ootomo CUBLAS",
        .description = "Like Ootomo v2 but without a merged kernel and 3 separate CUBLAS invocations",
        .highestPerforming = true,
    },
    {
        .function = matmul_cuda<float, float, 1, false>,
        .name = "matmul CUDA cores v1",
        .description = "CUDA core fp32 matrix multiplication with warptiling",
        .highestPerforming = true,
    },
    {
        .function = matmul_cuBLAS32,
        .name = "matmul_cuBLAS 32",
        .description = "cuBLAS",
        .highestPerforming = true,
    },
    {
        .function = matmul_ozaki_float<0>,
        .name = "Matmul Ozaki float v0",
        .description = "Ozaki FP32 using FP16.",
        .highestPerforming = true,
    },
    {
        .function = matmul_ozaki_float<1>,
        .name = "Matmul Ozaki float v1",
        .description = "Ozaki FP32 using FP16.",
        .highestPerforming = false,
    },
    {
        .function = matmul_ozaki_float<2>,
        .name = "Matmul Ozaki float v2",
        .description = "Ozaki FP32 using FP16.",
        .highestPerforming = false,
    },
};

matmul_variant<double> matmulVariants64[] =
{
    {
        .function = matmul_cuBLAS64,
        .name = "matmul_cuBLAS 64",
        .description = "cuBLAS",
        .highestPerforming = true,
    },
    // {
    //     .function = matmul_ozaki<0>,
    //     .name = "Matmul Ozaki v0",
    //     .description = "Ozaki FP64 using FP32 on CPU",
    //     .highestPerforming = false,
    // },
    {
        .function = matmul_ozaki<1>,
        .name = "Matmul Ozaki v1",
        .description = "Ozaki FP64 using FP32. Matmul on GPU, split-merge on CPU",
        .highestPerforming = false,
    },
/*
    {
        .function = matmul_ozaki<2>,
        .name = "Matmul Ozaki v2",
        .description = "Ozaki FP64 using FP16. Matmul on GPU, split-merge on CPU",
        .highestPerforming = true,
    },
    */
    {
        .function = matmul_ozaki<3>,
        .name = "Matmul Ozaki v3",
        .description = "Ozaki FP64 using FP32 4 splits and 16 matmuls",
        .highestPerforming = true,
    },
    {
        .function = matmul_ozaki<4>,
        .name = "Matmul Ozaki float 25 terms",
        .description = "Ozaki FP64 using FP32 5 splits and 25 matmuls",
        .highestPerforming = true,
    },
    {
        .function = matmul_ozaki<5>,
        .name = "Matmul Ozaki float 36 terms",
        .description = "Ozaki FP64 using FP32 6 splits and 36 matmuls",
        .highestPerforming = true,
    },
    {
        .function = matmul_ozaki<6>,
        .name = "Matmul Ozaki float, 4 split, 10 terms",
        .description = "Ozaki FP64 using FP32 4 splits and 10 matmuls",
        .highestPerforming = false,
    },
    {
        .function = matmul_ozaki<7>,
        .name = "Matmul Ozaki float, 5 split, 20 terms",
        .description = "Ozaki FP64 using FP32 5 splits and 20 matmuls",
        .highestPerforming = false,
    },
    {
        .function = matmul_ozaki<8>,
        .name = "Matmul Ozaki float 25 terms",
        .description = "Ozaki FP64 using FP32 6 splits and 25 matmuls",
        .highestPerforming = false,
    },
    {
        .function = matmul_ozaki<9>,
        .name = "Matmul Ozaki half 36 terms",
        .description = "Ozaki FP64 using FP16 6 splits and 36 matmuls",
        .highestPerforming = false,
    },
    {
        .function = matmul_ozaki<10>,
        .name = "Ozaki v10 FP32->FP64, 4spl 16mm, cuBLAS",
        .description = "Ozaki v10 FP32->FP64, 4spl 16mm, cuBLAS",
        .highestPerforming = true,
    },
    {
        .function = matmul_ozaki<11>,
        .name = "Ozaki v11 FP32->FP64, 5spl 25mm, cuBLAS",
        .description = "Ozaki v11 FP32->FP64, 5spl 25mm, cuBLAS",
        .highestPerforming = true,
    },
    {
        .function = matmul_ozaki<12>,
        .name = "Ozaki v12 FP32->FP64, 4spl 16mm, Ootomo",
        .description = "Ozaki v12 FP32->FP64, 4spl 16mm, Ootomo",
        .highestPerforming = true,
    },
    {
        .function = matmul_ozaki<13>,
        .name = "Ozaki v13 FP32->FP64, 5spl 25mm, Ootomo",
        .description = "Ozaki v13 FP32->FP64, 5spl 25mm, Ootomo",
        .highestPerforming = true,
    },
    {
        .function = matmul_Ootomo_double_v0,
        .name = "Ootomo double v0",
        .description = "Use external split to partition double into 4 float multiplications. Perform this 4 float multiplications with fp32 Ootomo. Merge the 4 results.",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis_double<0>,
        .name = "Simple Markidis double v0",
        .description = "Split 3 half, 9 float multiply & accumulate (TENSOR CORES)",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis_double<1>,
        .name = "Markidis double: 16 terms, float acc",
        .description = "Split 4 half, 16 float multimply & accumulate (TENSOR CORES)",
        .highestPerforming = false,      
    },
    {
        .function = matmul_simpleMarkidis_double<2>,
        .name = "Simple Markidis double v2",
        .description = "Split 4 half, 16 float multiply & double accumulate (CUDA CORES)",
        .highestPerforming = false,      
    },
    {
        .function = matmul_simpleMarkidis_double<3>,
        .name = "Markidis double: 16 terms, double acc",
        .description = "Split 4 half with scaling, 16 float multiply & double accumulate (CUDA CORES)",
        .highestPerforming = false,      
    },
    {
        .function = matmul_simpleMarkidis_double<4>,
        .name = "Markidis double: 25 terms, double acc",
        .description = "Split 5 half with scaling, 25 float multiply & double accumulate (CUDA CORES)",
        .highestPerforming = false,      
    },
    {
        .function = matmul_simpleMarkidis_double<5>,
        .name = "Markidis double: 36 terms, double acc",
        .description = "Split 6 half with scaling, 36 float multiply & double accumulate (CUDA CORES)",
        .highestPerforming = false,      
    },
    {
        .function = matmul_simpleMarkidis_double<6>,
        .name = "Hybrid double: 16 terms, 3 double & 13 float acc",
        .description = "Split 4 half, 16 float multiply & 13 float accumulate & 3 most significant terms double accumulate (CUDA & TENSOR CORES)",
        .highestPerforming = false,      
    },
    {
        .function = matmul_simpleMarkidis_double<7>,
        .name = "Hybrid double: 16 terms, 6 double & 10 float acc",
        .description = "Split 4 half, 16 float multiply & 10 float accumulate & 6 most significant terms double accumulate (CUDA & TENSOR CORES)",
        .highestPerforming = true,      
    },
    {
        .function = matmul_simpleMarkidis_double<8>,
        .name = "Hybrid double: 25 terms, 6 double & 19 float acc",
        .description = "Split 5 half, 25 float multiply & 19 float accumulate & 6 most significant terms double accumulate (CUDA & TENSOR CORES)",
        .highestPerforming = true,      
    },
    {
        .function = matmul_simpleMarkidis_double<9>,
        .name = "Simple Markidis double v9",
        .description = "Split 2 float, 4 float multiply & accumulate (CUDA CORES)",
        .highestPerforming = false,      
    },
    {
        .function = matmul_simpleMarkidis_double<10>,
        .name = "Simple Markidis double v10",
        .description = "Split 2 float, 4 float multiply & double accumulate (CUDA CORES)",
        .highestPerforming = false,      
    },
    {
        .function = matmul_simpleMarkidis_double<11>,
        .name = "Simple Markidis double v11",
        .description = "Split 2 float with scaling, 4 float multiply & double accumulate (CUDA CORES)",
        .highestPerforming = false,      
    },
    {
        .function = matmul_simpleMarkidis_double<12>,
        .name = "Simple Markidis double v12",
        .description = "Split 2 float, 4 double multiply & accumulate (CUDA CORES)",
        .highestPerforming = false,      
    },
    {
        .function = matmul_simpleMarkidis_double<13>,
        .name = "Simple Markidis double v13",
        .description = "Split 5 half with scaling, 25 half->float multiply & double accumulate (TENSOR CORES)",
        .highestPerforming = false,      
    },
    {
        .function = matmul_simpleMarkidis_double<14>,
        .name = "Simple Markidis double v14",
        .description = "Split 4 half with scaling, 3 float/double, 13 half->float/double multiply/accumulate (CUDA CORES, TENSOR CORES)",
        .highestPerforming = false,      
    },
    {
        .function = matmul_cuda<double, double, 1, false>,
        .name = "matmul CUDA cores v1",
        .description = "double matmul using CUDA cores",
        .highestPerforming = true,
    },
#if SM_VERSION >= 800
    {
        .function = matmul_simpleMarkidis_double_double<0>,
        .name = "Simple Markidis double double v0",
        .description = "Split 3, 9 multiply",
        .highestPerforming = false,
    },
    {
        .function = matmul_simpleMarkidis_double_double<1>,
        .name = "Simple Markidis double double v1",
        .description = "Split 4, 16 multiply",
        .highestPerforming = false,
    },
    {
        .function = matmul_cuda<double, double, 0, true>,
        .name = "matmul_tensor v0",
        .description = "simple tensor matmul",
        .highestPerforming = false,
    },
    {
        .function = matmul_cuda<double, double, 1, true>,
        .name = "matmul_tensor v1",
        .description = "multiple warps per block",
        .highestPerforming = false,
    },
    {
        .function = matmul_cuda<double, double, 2, true>,
        .name = "matmul_tensor v2",
        .description = "with shared memory",
        .highestPerforming = false,
    },
    {
        .function = matmul_cuda<double, double, 3, true>,
        .name = "matmul_tensor v3",
        .description = "with shared memory",
        .highestPerforming = false,
    }
#endif
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
                std::cout << ", ";
            std::cout << A[m*N + n];
        }
        std::cout << std::endl;
    }
}

template<class T>
void testMatmulCorrectness(matmul_variant<T>* function, LCG *rng)
{
    // Parameters
    // srand(time(NULL));
    const size_t RUNS = 10;
    const size_t NUM_TYPES = 5;
    bool failed = false;
    double *residual_sums = (double*) calloc(NUM_TYPES, sizeof(double));
    for (size_t run = 0; run < RUNS; run++) {
        uint64_t starting_seed = rng->state;
        // Randomize matrix dimensions
        size_t M, K, N;
        M = 1 << next_int(rng, 7, 9);
        K = 1 << next_int(rng, 7, 9);
        N = 1 << next_int(rng, 7, 9);

        // Get matrices
        T *C;
        PRINT_ON_ERROR(cudaMallocHost(&C, M * N * sizeof(T)));

        for (size_t input_type = 0; input_type < NUM_TYPES; input_type++) {
            auto [A, B, C_reference] = getMatrices<T>(M, K, N, input_type, 0, rng);

            // Call function under test
            function->function(A, B, C, M, K, N);

            // Calculate error
            double relative_residual = rel_residual(C, C_reference, M * N);
            double absolute_residual = abs_residual(C, C_reference, M * N);
            residual_sums[input_type] += relative_residual;
            
            // Analyze result for "regular" inputs
            if (input_type < 2) {
                int wrong = test_matmul_correctness_full(C, C_reference, M, N);
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
                    std::cout << "\t" << "Input type: \033[33m" << input_type << "\033[0m" << std::endl;
                    std::cout << "\t" << "Wrong at: \033[33mRow " << row << "\033[0m, \033[33mCol " << col << "\033[0m" << std::endl;
                    std::cout << "\t" << "Expected: \033[33m" << ref_sol << "\033[0m\tActual:   \033[33m" << wrong_val << "\033[0m" << std::endl;
                    std::cout << "\t" << "Error:    \033[33m" << rel_err << "\033[0m (rel) \033[33m" << abs_err << "\033[0m (abs)" << std::endl;

                }
            }
            PRINT_ON_ERROR(cudaFreeHost(A));
            PRINT_ON_ERROR(cudaFreeHost(B));
            PRINT_ON_ERROR(cudaFreeHost(C_reference));

            if (failed) {
                break;
            }
        }

        // Free memory
        PRINT_ON_ERROR(cudaFreeHost(C));

        if(failed) break;
    }

    if (!failed) {
        // Success!
        std::cout
            << "\033[32m" << "[SUCCESS]  " << "\033[0m" // Green text
            << std::left << std::setw(FUNCTION_NAME_WIDTH) << function->name;
        // Print residual errors
        
        std::cout << "Relative residuals: ";
        for (size_t input_type = 0; input_type < NUM_TYPES; input_type++) {
            double avg_rel_residual = residual_sums[input_type] / (double) RUNS;
            std::cout << "Type " << input_type << ": \033[33m" << std::left << std::setw(11) << avg_rel_residual << "\033[0m ";
        }
        std::cout << std::endl;
    }
}

template<class T>
void profile(matmul_variant<T> variant, int warmup, int iterations, size_t M, size_t K, size_t N)
{
    T *A = (T*)calloc(M * K, sizeof(*A));
    T *B = (T*)calloc(K * N, sizeof(*B));
    T *C = (T*)calloc(M * N, sizeof(*C));
    LCG rng = rng_seeded(0xC0FEE);
    fill_matrices(&rng, 0, A, B, M*K, K*N);

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

void testTranspose()
{
    const int width = 1024;
    const int height = 2048;
    const int size = width * height;

    float *A, *A_T, *A_TT;

    // Allocate pinned host memory
    PRINT_ON_ERROR(cudaMallocHost(&A, size * sizeof(float)));
    PRINT_ON_ERROR(cudaMallocHost(&A_T, size * sizeof(float)));
    PRINT_ON_ERROR(cudaMallocHost(&A_TT, size * sizeof(float)));

    // Initialize input matrix
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            A[i * width + j] = static_cast<float>(i * width + j);
        }
    }

    transposeMatrix(A, A_T, height, width);
    transposeMatrix(A_T, A_TT, width, height);

    for (int i = 0; i < size; i++)
    {
        if (A[i] != A_TT[i])
            printf("Error transposing");
    }

    PRINT_ON_ERROR(cudaFreeHost(A));
    PRINT_ON_ERROR(cudaFreeHost(A_T));
    PRINT_ON_ERROR(cudaFreeHost(A_TT));
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

        // Set epsilon to e.g. 1e-9 to see some bits
        //test_ozaki_split_correctness(&rng, 1e-9, 10, false);
        
        //profile(matmulVariants64[0], 0, 1, 8192, 8192, 8192);
        //profile(matmulVariants64[1], 0, 1, 8192, 8192, 8192);
        //profile(matmulVariants64[3], 0, 1, 8192, 8192, 8192);
        /*        
        profile(matmulVariants64[2], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants64[3], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants64[4], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants64[5], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants64[4], 0, 1, 4096, 4096, 4096);
        */
        // profile(matmulVariants64[7], 0, 1, 4096, 4096, 4096);
        // profile(matmulVariants64[14], 0, 1, 4096, 4096, 4096);
        // profile(matmulVariants64[15], 0, 1, 4096, 4096, 4096);
        
        // profile(matmulVariants32[3], 1, 5, 8192, 8192, 8192);
        // profile(matmulVariants32[4], 1, 5, 8192, 8192, 8192);
        /*
        profile(matmulVariants32[1], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants32[2], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants32[4], 0, 1, 8192, 8192, 8192);
        profile(matmulVariants32[5], 0, 1, 8192, 8192, 8192);
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

        if (strcmp(argv[1], "32") == 0)
        {
            std::vector<int> timeIndices32;
            for(size_t i = 0; i < ARRAY_COUNT(matmulVariants32); i++) {
                if (matmulVariants32[i].highestPerforming) {
                    timeIndices32.push_back(i);
                }
            }

            // timeIndices32 = {3, 4, 5, 6};
            for(const int value : timeIndices32)
            {
                timeFunction(&matmulVariants32[value], argv[3], rng);
                rng.state = seed;
            }
        } else if (strcmp(argv[1], "64") == 0) {
            std::vector<int> timeIndices64;
            for(size_t i = 0; i < ARRAY_COUNT(matmulVariants64); i++) {
                if (matmulVariants64[i].highestPerforming) {
                    timeIndices64.push_back(i);
                }
            }

            // timeIndices64 = {};
            for(const int value : timeIndices64)
            {
                timeFunction(&matmulVariants64[value], argv[3], rng);
                rng.state = seed;
            }
        } else {
            printf("Usage: %s 32|64\n", argv[0]);
        }

        
        
    }
    return 0;
}
