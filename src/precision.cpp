#include "precision.h"

template<class T>
double frobenius_norm(T *A, int n) {
    __float80 sqr_sum = 0.0;
    for(int i = 0; i < n; i++) {
        __float80 a_i = (__float80) A[i];
        sqr_sum += a_i * a_i;
    }
    return (double) sqrt(sqr_sum);
}

template<class T>
double abs_residual(T *result, T *reference, int n) {
    __float80 sqr_sum_err = 0.0;
    for(int i = 0; i < n; i++) {
        __float80 ref = (__float80) reference[i];
        __float80 err = ref - ((__float80) result[i]);
        sqr_sum_err += err * err;
    }
    return (double) sqrt(sqr_sum_err);
}

template<class T>
double rel_residual(T *result, T *reference, int n) {
    __float80 sqr_sum_err = 0.0;
    __float80 sqr_sum_ref = 0.0;
    for(int i = 0; i < n; i++) {
        __float80 ref = (__float80) reference[i];
        __float80 err = ref - ((__float80) result[i]);
        sqr_sum_err += err * err;
        sqr_sum_ref += ref * ref;
    }
    return (double) (sqrt(sqr_sum_err / sqr_sum_ref));
}

template<class T>
double capped_rel_err(T *result, T *reference, int n) {
    // Smallest type 4 float squared
    const __float80 EPS = (__float80) construct_double(false, -200, 0);

    const __float80 ONE = 1.0;
    __float80 sum_rel_err = 0.0;
    for(int i = 0; i < n; i++) {
        __float80 ref = (__float80) reference[i];
        __float80 err = ref - ((__float80) result[i]);

        __float80 rel_err;
        // Check this way so NAN enters the fallback branch
        if (!(abs(ref) > EPS)) {
            rel_err = abs(err / EPS);
        } else {
            rel_err = abs(err / ref);
        }
        // Check this way so NAN enters the fallback branch
        if (!(rel_err >= 0) || !(rel_err <= 1)) {
            rel_err = ONE;
        }
        sum_rel_err += rel_err;
    }

    __float80 inv_n = ONE / ((__float80) n);
    double res = (double) (inv_n * sum_rel_err);
    if (!isfinite(res)) {
        printf("Warning: non-finite precision metric!\n");
    }
    return res;
}

template double frobenius_norm<float>(float *result, int n);
template double frobenius_norm<double>(double *result, int n);
template double abs_residual<float>(float *result, float *reference, int n);
template double abs_residual<double>(double *result, double *reference, int n);
template double rel_residual<float>(float *result, float *reference, int n);
template double rel_residual<double>(double *result, double *reference, int n);
template double capped_rel_err<float>(float *result, float *reference, int n);
template double capped_rel_err<double>(double *result, double *reference, int n);

template<class T>
void calc_precision_metrics(T *result, T *reference, int n, double *metrics) {
    // Smallest type 4 float squared
    const __float80 EPS = (__float80) construct_double(false, -200, 0);
    __float80 sum_abs_ref = 0.0;
    __float80 sum_abs_err = 0.0;

    __float80 sum_rel_err = 0.0;
    __float80 sum_rel_max = 0.0;
    __float80 sum_rel_adj = 0.0;
    __float80 sum_rel_min1 = 0.0;

    __float80 sum_sqr_ref = 0.0;
    __float80 sum_sqr_err = 0.0;
    __float80 sum_sqr_rel = 0.0;

    __float80 sum_log_err = 0.0;
    for(int i = 0; i < n; i++) {
        __float80 res = (__float80) result[i];
        __float80 ref = (__float80) reference[i];

        __float80 abs_err = abs(ref - res);
        __float80 rel_err = abs_err / std::max(abs(ref), EPS);
        __float80 rel_err_max = abs_err / std::max(abs(ref), abs(res));
        __float80 rel_err_adj = abs_err / std::max(abs(ref + ref - res), abs(res));
        __float80 rel_err_min1 = std::min(rel_err, (__float80) 1.0);

        __float80 sqr_ref = ref * ref;
        __float80 sqr_err = abs_err * abs_err;
        __float80 sqr_rel = rel_err * rel_err;

        __float80 log_err = (abs(res) <= 0.0 || abs(ref) <= 0.0) ? 1.0 : abs(std::log2(abs(res)) - std::log2(abs(ref)));

        sum_abs_err += abs_err;
        sum_abs_ref += abs(ref);

        sum_rel_err += rel_err;
        sum_rel_max += rel_err_max;
        sum_rel_adj += rel_err_adj;
        sum_rel_min1 += rel_err_min1;

        sum_sqr_ref += sqr_ref;
        sum_sqr_err += sqr_err;
        sum_sqr_rel += sqr_rel;

        sum_log_err += log_err;
    }
    
    __float80 inv_n = ((__float80) 1.0) / ((__float80) n);
    
    __float80 residual = std::sqrt(sum_sqr_err / sum_sqr_ref);
    __float80 residual_l1 = sum_abs_err / sum_abs_ref;

    __float80 mean_abs_err = sum_abs_err * inv_n;
    __float80 mean_sqr_err = std::sqrt(sum_sqr_err * inv_n);

    __float80 mean_rel_err = sum_rel_err * inv_n;
    __float80 mean_rel_sqr = std::sqrt(sum_sqr_rel * inv_n);
    __float80 mean_rel_max = sum_rel_max * inv_n;
    __float80 mean_rel_adj = sum_rel_adj * inv_n;
    __float80 mean_rel_min1 = sum_rel_min1 * inv_n;

    __float80 mean_log_err = sum_log_err * inv_n;

    metrics[0] = (double) residual;
    metrics[1] = (double) residual_l1;
    metrics[2] = (double) mean_abs_err;
    metrics[3] = (double) mean_sqr_err;
    metrics[4] = (double) mean_rel_err;
    metrics[5] = (double) mean_rel_sqr;
    metrics[6] = (double) mean_rel_max;
    metrics[7] = (double) mean_rel_adj;
    metrics[8] = (double) mean_rel_min1;
    metrics[9] = (double) mean_log_err;
}

template void calc_precision_metrics<float>(float *result, float *reference, int n, double *metrics);
template void calc_precision_metrics<double>(double *result, double *reference, int n, double *metrics);

/// Returns the index of an element with error that is too great. If none is found, returns -1
template<class T>
int test_matmul_correctness_probabilistic(LCG *rng, T *A, T *B, T *C, size_t M, size_t K, size_t N) {
    // Test the square root of the matrix size
    size_t num_to_test = std::max((size_t) 100, (size_t) sqrt(M * N));
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
