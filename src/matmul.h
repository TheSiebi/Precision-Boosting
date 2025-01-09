#ifndef MATMUL_H
#define MATMUL_H

#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "rand.h"
#include "cuda_fp16.h"

flop_counts matmul_reference32(float *A, float *B, float *C, size_t M, size_t K, size_t N);
flop_counts matmul_reference64(double *A, double *B, double *C, size_t M, size_t K, size_t N);

flop_counts matmul_v0(double *A, double *B, double *C, size_t M, size_t K, size_t N);
// flop_counts matmul_v1(double *A, double *B, double *C, size_t M, size_t K, size_t N);
// flop_counts matmul_v2(double *A, double *B, double *C, size_t M, size_t K, size_t N);
// flop_counts matmul_v3(double *A, double *B, double *C, size_t M, size_t K, size_t N);

template<typename InpuType, typename OutputType, int version, bool useTensorCores>
flop_counts matmul_cuda(InpuType *A, InpuType *B, OutputType *C, size_t M, size_t K, size_t N);

template<int version, int streamCount, bool useScale>
flop_counts matmul_simpleMarkidis(float *A, float *B, float *C, size_t M, size_t K, size_t N); 
flop_counts matmul_markidis(float *A, float *B, float *C, size_t M, size_t K, size_t N);

template<int version>
flop_counts matmul_simpleMarkidis_float(float *A, float *B, float *C, size_t M, size_t K, size_t N);
template<int version>
flop_counts matmul_simpleMarkidis_double(double *A, double *B, double *C, size_t M, size_t K, size_t N);

template<int version>
flop_counts matmul_simpleMarkidis_double_double(double *A, double *B, double *C, size_t M, size_t K, size_t N);

flop_counts matmul_basic_Ootomo_v0(float *A, float *B, float *C, size_t M, size_t K, size_t N);

flop_counts matmul_Ootomo_v0(float *A, float *B, float *C, size_t M, size_t K, size_t N);
flop_counts matmul_Ootomo_v1(float *A, float *B, float *C, size_t M, size_t K, size_t N);
flop_counts matmul_Ootomo_v2(float *A, float *B, float *C, size_t M, size_t K, size_t N);
flop_counts matmul_Ootomo_v3(float *A, float *B, float *C, size_t M, size_t K, size_t N);
flop_counts matmul_Ootomo_double_v0(double *A, double *B, double *C, size_t M, size_t K, size_t N);

flop_counts matmul_cuBLAS32(float *A, float *B, float *C, size_t M, size_t K, size_t N);
flop_counts matmul_cuBLAS64(double *A, double *B, double *C, size_t M, size_t K, size_t N);
flop_counts matmul_cuBLASMixed(half *A, half *B, float *C, size_t M, size_t K, size_t N);

template <typename T>
void transposeMatrix(T *A, T *A_T, size_t M, size_t K);
// Ozaki paper uses A [m, n] and B [n, p] matrices
void test_ozaki_split_correctness(LCG* rng, const double epsilon, const size_t max_splits, const bool verbose);
template<int version>
flop_counts matmul_ozaki(double *a, double *b, double *c, size_t m, size_t n, size_t p);

template<int version>
flop_counts matmul_ozaki_float(float *a, float *b, float *c, size_t m, size_t n, size_t p);

#endif // MATMUL_H
