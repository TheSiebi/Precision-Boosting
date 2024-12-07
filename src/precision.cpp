#include "precision.h"

template<class T>
double frobenius_norm(T *A, int n) {
    double sqr_sum = 0.0;
    for(int i = 0; i < n; i++) {
        double a_i = (double) A[i];
        sqr_sum += a_i * a_i;
    }
    double result = sqrt(sqr_sum);
    return result;
}

template<class T>
double abs_residual(T *result, T *reference, int n) {
    double sqr_sum_err = 0.0;
    for(int i = 0; i < n; i++) {
        double ref = (double) reference[i];
        double err = ref - ((double) result[i]);
        sqr_sum_err += err * err;
    }
    return sqrt(sqr_sum_err);
}

template<class T>
double rel_residual(T *result, T *reference, int n) {
    double sqr_sum_err = 0.0;
    double sqr_sum_ref = 0.0;
    for(int i = 0; i < n; i++) {
        double ref = (double) reference[i];
        double err = ref - ((double) result[i]);
        sqr_sum_err += err * err;
        sqr_sum_ref += ref * ref;
    }
    return sqrt(sqr_sum_err / sqr_sum_ref);
}

template double frobenius_norm<float>(float *result, int n);
template double frobenius_norm<double>(double *result, int n);
template double abs_residual<float>(float *result, float *reference, int n);
template double abs_residual<double>(double *result, double *reference, int n);
template double rel_residual<float>(float *result, float *reference, int n);
template double rel_residual<double>(double *result, double *reference, int n);

/// Returns the index of an element with error that is too great. If none is found, returns -1
template<class T>
int test_matmul_correctness_probabilistic(LCG *rng, T *A, T *B, T *C, size_t M, size_t K, size_t N) {
    // Test the square root of the matrix size
    size_t num_to_test = 1e6 + sqrt(M * N - 1e6);
    while (num_to_test--) {
        // Generate random index and test
        size_t index = next_below(rng, (uint32_t) (M * N));
        if (!test_matmul_correctness_single<T>(A, B, C, K, N, index)) {
            return index;
        }
    }

    return -1;
}

/// Returns the index of an element with error that is too great. If none is found, returns -1
template<class T>
int test_matmul_correctness_full(T *C, T *C_reference, size_t M, size_t N) {
    for (size_t index = 0; index < M * N; index++) {
        // Calculate error
        double abs_err_ij = abs(C_reference[index] - C[index]);
        double rel_err_ij = abs_err_ij / abs(C_reference[index]); 
        // Check cut-off bounds
        if (abs_err_ij > MAX_ABSOLUTE_ERROR && rel_err_ij > MAX_RELATIVE_ERROR) {
            return index;
        }
    }
    return -1;
}

/// Returns true if the error is within acceptable bounds
template<class T>
bool test_matmul_correctness_single(T *A, T *B, T *C, size_t K, size_t N, size_t index) {
    // Reference solution in double precision
    double c_ij = referenceMatmul_element(A, B, K, N, index);
    // Calculate error
    double abs_err_ij = abs(c_ij - C[index]);
    double rel_err_ij = abs_err_ij / abs(c_ij); 
    // Check cut-off bounds
    return abs_err_ij <= MAX_ABSOLUTE_ERROR || rel_err_ij <= MAX_RELATIVE_ERROR;
}

template<class T>
void referenceMatmul_full(T *A, T *B, T *C, size_t M, size_t K, size_t N) {
    for (size_t index = 0; index < M * N; index++) {
        C[index] = (T) referenceMatmul_element(A, B, K, N, index);
    }
}

template<class T>
double referenceMatmul_element(T *A, T *B, size_t K, size_t N, size_t index) {
    size_t i = index / N;
    size_t j = index % N;
    __float80 c_ij = 0.0;
    for (size_t k = 0; k < K; k++) {
        __float80 a_ik = (__float80) A[i*K + k];
        __float80 b_kj = (__float80) B[k*N + j];
        c_ij += a_ik * b_kj;
    }
    return (double) c_ij;
}

template int test_matmul_correctness_probabilistic<float>(LCG *rng, float *A, float *B, float *C, size_t M, size_t K, size_t N);
template int test_matmul_correctness_probabilistic<double>(LCG *rng, double *A, double *B, double *C, size_t M, size_t K, size_t N);

template int test_matmul_correctness_full<float>(float *C, float *C_reference, size_t M, size_t N);
template int test_matmul_correctness_full<double>(double *C, double *C_reference, size_t M, size_t N);

template double referenceMatmul_element<float>(float *A, float *B, size_t K, size_t N, size_t index);
template double referenceMatmul_element<double>(double *A, double *B, size_t K, size_t N, size_t index);

template void referenceMatmul_full<float>(float *A, float *B, float *C, size_t M, size_t K, size_t N);
template void referenceMatmul_full<double>(double *A, double *B, double *C, size_t M, size_t K, size_t N);

template<class T>
bool test_equality(T *A, T *B, size_t N) {
    for (size_t i = 0; i < N; i++) {
        double expected = B[i];
        double actual = A[i];
        // Calculate error
        double abs_err_ij = abs(expected - actual);
        double rel_err_ij = abs_err_ij / abs(expected);
        // Check cut-off bounds
        if (abs_err_ij > MAX_ABSOLUTE_ERROR && rel_err_ij > MAX_RELATIVE_ERROR) {
            return false;
        }
    }
    // All errors within acceptable range
    return true;
}

template bool test_equality<float>(float *A, float *B, size_t N);
template bool test_equality<double>(double *A, double *B, size_t N);
