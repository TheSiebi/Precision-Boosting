#ifndef PRECISION_H
#define PRECISION_H

#include "rand.h"
#include <math.h>

const double MAX_RELATIVE_ERROR = 1e-1; // A relative error above this value will count as a fail
const double MAX_ABSOLUTE_ERROR = 1e-4; // Any error less than this will not count as a failure, even if relative error is too large

template<class T>
double frobenius_norm(T *A, int n);

template<class T>
double abs_residual(T *result, T *reference, int n);

template<class T>
double rel_residual(T *result, T *reference, int n);

template<class T>
double capped_rel_err(T *result, T *reference, int n);

extern template double frobenius_norm<float>(float *result, int n);
extern template double frobenius_norm<double>(double *result, int n);
extern template double abs_residual<float>(float *result, float *reference, int n);
extern template double abs_residual<double>(double *result, double *reference, int n);
extern template double rel_residual<float>(float *result, float *reference, int n);
extern template double rel_residual<double>(double *result, double *reference, int n);
extern template double capped_rel_err<float>(float *result, float *reference, int n);
extern template double capped_rel_err<double>(double *result, double *reference, int n);

template<class T>
void calc_precision_metrics(T *result, T *reference, int n, double *metrics);

extern template void calc_precision_metrics<float>(float *result, float *reference, int n, double *metrics);
extern template void calc_precision_metrics<double>(double *result, double *reference, int n, double *metrics);

template<class T>
int test_matmul_correctness_probabilistic(LCG *rng, T *A, T *B, T *C, size_t M, size_t K, size_t N);

template<class T>
int test_matmul_correctness_full(T *C, T *C_reference, size_t M, size_t N);

template<class T>
bool test_matmul_correctness_single(T *A, T *B, T *C, size_t K, size_t N, size_t index);

template<class T>
void referenceMatmul_full(T *A, T *B, T *C, size_t M, size_t K, size_t N);

template<class T>
double referenceMatmul_element(T *A, T *B, size_t K, size_t N, size_t index);

extern template int test_matmul_correctness_probabilistic<float>(LCG *rng, float *A, float *B, float *C, size_t M, size_t K, size_t N);
extern template int test_matmul_correctness_probabilistic<double>(LCG *rng, double *A, double *B, double *C, size_t M, size_t K, size_t N);

extern template int test_matmul_correctness_full<float>(float *C, float *C_reference, size_t M, size_t N);
extern template int test_matmul_correctness_full<double>(double *C, double *C_reference, size_t M, size_t N);

extern template double referenceMatmul_element<float>(float *A, float *B, size_t K, size_t N, size_t index);
extern template double referenceMatmul_element<double>(double *A, double *B, size_t K, size_t N, size_t index);

extern template void referenceMatmul_full<float>(float *A, float *B, float *C, size_t M, size_t K, size_t N);
extern template void referenceMatmul_full<double>(double *A, double *B, double *C, size_t M, size_t K, size_t N);

template<class T>
bool test_equality(T *A, T *B, size_t N);

extern template bool test_equality<float>(float *A, float *B, size_t N);
extern template bool test_equality<double>(double *A, double *B, size_t N);

#endif // PRECISION_H
