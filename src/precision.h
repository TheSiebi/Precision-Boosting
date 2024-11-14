#ifndef PRECISION_H
#define PRECISION_H

#include "rand.h"
#include <math.h>

template<class T>
double frobenius_norm(T *A, int n);

template<class T>
double abs_residual(T *result, double *reference, int n);

template<class T>
double rel_residual(T *result, double *reference, int n);

extern template double frobenius_norm<float>(float *result, int n);
extern template double frobenius_norm<double>(double *result, int n);
extern template double abs_residual<float>(float *result, double *reference, int n);
extern template double abs_residual<double>(double *result, double *reference, int n);
extern template double rel_residual<float>(float *result, double *reference, int n);
extern template double rel_residual<double>(double *result, double *reference, int n);

template<class T>
int test_matmul_correctness_probabilistic(LCG *rng, T *A, T *B, T *C, size_t M, size_t K, size_t N);

template<class T>
int test_matmul_correctness_full(T *A, T *B, T *C, size_t M, size_t K, size_t N);

template<class T>
bool test_matmul_correctness_single(T *A, T *B, T *C, size_t M, size_t K, size_t N, size_t index);

template<class T>
void referenceMatmul_full(T *A, T *B, T *C, int M, int K, int N);

template<class T>
double referenceMatmul_element(T *A, T *B, size_t K, size_t N, size_t index);

extern template int test_matmul_correctness_probabilistic<float>(LCG *rng, float *A, float *B, float *C, size_t M, size_t K, size_t N);
extern template int test_matmul_correctness_probabilistic<double>(LCG *rng, double *A, double *B, double *C, size_t M, size_t K, size_t N);

extern template int test_matmul_correctness_full<float>(float *A, float *B, float *C, size_t M, size_t K, size_t N);
extern template int test_matmul_correctness_full<double>(double *A, double *B, double *C, size_t M, size_t K, size_t N);

extern template double referenceMatmul_element<float>(float *A, float *B, size_t K, size_t N, size_t index);
extern template double referenceMatmul_element<double>(double *A, double *B, size_t K, size_t N, size_t index);

#endif // PRECISION_H