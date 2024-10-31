#ifndef FUNCTIONTIMER_H
#define FUNCTIONTIMER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>

typedef void (*MatMul)(double *A, double *B, double *C, int M, int K, int N);
struct matmul_variant
{
    MatMul function;
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
    int M;
    int N;
    int K;
    double *timings;
};

struct measurement
{
    struct measurementConfiguration configuration;
    struct run *runs;
};
void timeRun(double *timings, int iterations, int M, int K, int N, MatMul func);
void timeFunction(struct matmul_variant *function, char *path);


#endif // FUNCTIONTIMER_H
