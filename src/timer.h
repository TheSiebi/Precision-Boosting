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
    size_t flops16;
    size_t flops32;
    size_t flops64;
};

template<class T>
using MatMul = flop_counts (*)(T *A, T *B, T *C, size_t M, size_t K, size_t N);

template<class T>
struct matmul_variant
{
    MatMul<T> function;
    const char *name;
    const char *description;
    const bool highestPerforming;
};

typedef void (*Split)(const double *A, void *A16, void *dA16, size_t M, size_t N);
typedef void (*Merge)(const void *A16, const void *dA16, double* merged, size_t M, size_t N);
typedef void (*Splitf)(const float *A, void *A16, void *dA16, size_t M, size_t N);
typedef void (*Mergef)(const void *A16, const void *dA16, float* merged, size_t M, size_t N);

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
    size_t M;
    size_t K;
    size_t N;
    // Actually executed flops per precision (fp16 means on tensor core)
    size_t flops16;
    size_t flops32;
    size_t flops64;
    // Theoretical, mathematical number of flops
    size_t math_flops;
    double *timings;
    char* profile_output;
    bool sanity_check;
    struct precisionMeasurement *precMs;
};

struct measurement
{
    struct measurementConfiguration configuration;
    struct run *runs;
};

struct precisionMeasurement
{
    int input_type;
    double *residuals;
};

template<class T>
bool timeRun(double *timings, flop_counts *counts, int iterations, int input_type, int warmupIterations, size_t M, size_t K, size_t N, MatMul<T> func, LCG rng);

template<class T>
void timeFunction(struct matmul_variant<T> *function, char *path, LCG rng);

flop_counts matmul_flopcount_32(size_t M, size_t K, size_t N);
flop_counts matmul_flopcount_64(size_t M, size_t K, size_t N);


#endif // FUNCTIONTIMER_H
