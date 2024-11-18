#ifndef MATMUL_H
#define MATMUL_H

#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

flop_counts matmul_v0(double *A, double *B, double *C, int M, int K, int N);
// flop_counts matmul_v1(double *A, double *B, double *C, int M, int K, int N);
// flop_counts matmul_v2(double *A, double *B, double *C, int M, int K, int N);
// flop_counts matmul_v3(double *A, double *B, double *C, int M, int K, int N);

flop_counts matmul_cuda_v0(double *A, double *B, double *C, int M, int K, int N);
flop_counts matmul_simpleMarkidis_v0(float *A, float *B, float *C, int M, int K, int N);
flop_counts matmul_simpleMarkidis_v1(float *A, float *B, float *C, int M, int K, int N);
flop_counts matmul_simpleMarkidis_v2(float *A, float *B, float *C, int M, int K, int N);
flop_counts matmul_simpleMarkidis_v3(float *A, float *B, float *C, int M, int K, int N);
flop_counts matmul_simpleMarkidis_v4(float *A, float *B, float *C, int M, int K, int N);
flop_counts matmul_simpleOotomo_v0(float *A, float *B, float *C, int M, int K, int N);

flop_counts matmul_Ootomo_v0(float *A, float *B, float *C, int M, int K, int N);
flop_counts matmul_Ootomo_v1(float *A, float *B, float *C, int M, int K, int N);
flop_counts matmul_Ootomo_v2(float *A, float *B, float *C, int M, int K, int N);
flop_counts matmul_Ootomo_double_v0(double *A, double *B, double *C, int M, int K, int N);

flop_counts matmul_cuBLAS32(float *A, float *B, float *C, int M, int K, int N);
flop_counts matmul_cuBLAS64(double *A, double *B, double *C, int M, int K, int N);

// Ozaki paper uses A [m, n] and B [n, p] matrices
flop_counts matmul_Ozaki_v0(double *a, double *b, double *c, int m, int n, int p);

#endif // MATMUL_H
