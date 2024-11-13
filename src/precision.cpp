#include "precision.h"

#include <math.h>

template<class T>
T frobenius_norm(T *A, int n) {
    T sqr_sum = 0.0;
    for(int i = 0; i < n; i++) {
        T a_i = A[i];
        sqr_sum += a_i * a_i;
    }
    T result = sqrt(sqr_sum);
    return result;
}

double relative_residual(double *result, double *reference, int n) {
    double sqr_sum_err = 0.0;
    double sqr_sum_ref = 0.0;
    for(int i = 0; i < n; i++) {
        double ref = reference[i];
        double err = ref - result[i];
        sqr_sum_err += err * err;
        sqr_sum_ref += ref * ref;
    }
    return sqrt(sqr_sum_err / sqr_sum_ref);
}