#ifndef PRECISION_H
#define PRECISION_H

template<class T>
void frobenius_norm(T *A, int n);

double relative_residual(double *result, double *reference, int n);

#endif // PRECISION_H