#ifndef FUNCTIONTIMER_H
#define FUNCTIONTIMER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>
#include "rand.h"

struct flop_counts
{
    long flops16;
    long flops32;
    long flops64;
};

template<class T>
using MatMul = flop_counts (*)(T *A, T *B, T *C, int M, int K, int N);

template<class T>
struct matmul_variant
{
    MatMul<T> function;
    const char *name;
    const char *description;
};

typedef void (*Split)(const double *A, void *A16, void *dA16, int M, int N);
typedef void (*Merge)(const void *A16, const void *dA16, double* merged, int M, int N);
typedef void (*Splitf)(const float *A, void *A16, void *dA16, int M, int N);
typedef void (*Mergef)(const void *A16, const void *dA16, float* merged, int M, int N);

struct split_variant
{
    Split function;
    Splitf functionf;
    Merge invFunction;
    Mergef invFunctionf;
    const char *name;
    const char *description;
};


struct measurementConfiguration
{
    const char *timeStamp;
    const char *flags;
    const char *cpuModel;
    const char *gpuModel;
    const char *targetFunction;
};

struct run
{
    // Matrix dimensions
    int M;
    int N;
    int K;
    // Actually executed flops per precision (fp16 means on tensor core)
    long flops16;
    long flops32;
    long flops64;
    // Theoretical, mathematical number of flops
    long math_flops;
    double *timings;
};

struct measurement
{
    struct measurementConfiguration configuration;
    struct run *runs;
};

template<class T>
flop_counts timeRun(double *timings, int iterations, int M, int K, int N, MatMul<T> func, LCG rng);

template<class T>
void timeFunction(struct matmul_variant<T> *function, char *path, LCG rng);

flop_counts matmul_flopcount_32(int M, int K, int N);
flop_counts matmul_flopcount_64(int M, int K, int N);


#endif // FUNCTIONTIMER_H
