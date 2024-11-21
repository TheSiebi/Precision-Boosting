#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include "machine.h"
#include "../lib/cjson/cJSON.h"


#define NS_PER_SECOND 1e9

cJSON* measurement_configuration_to_json(struct measurementConfiguration *conf) {
    cJSON *json = cJSON_CreateObject();
    if (!json) return NULL;

    cJSON_AddStringToObject(json, "timeStamp", conf->timeStamp);
    cJSON_AddStringToObject(json, "flags", conf->flags);
    cJSON_AddStringToObject(json, "cpu model", conf->cpuModel);
    cJSON_AddStringToObject(json, "gpu model", conf->gpuModel);
    cJSON_AddStringToObject(json, "function name", conf->targetFunction);

    return json;
}

cJSON* run_to_json(struct run *r, int iterationsPerConfig) {
    cJSON *json = cJSON_CreateObject();
    if (!json) return NULL;

    cJSON_AddNumberToObject(json, "M", r->M);
    cJSON_AddNumberToObject(json, "K", r->K);
    cJSON_AddNumberToObject(json, "N", r->N);
    cJSON_AddNumberToObject(json, "flops16", r->flops16);
    cJSON_AddNumberToObject(json, "flops32", r->flops32);
    cJSON_AddNumberToObject(json, "flops64", r->flops64);
    cJSON_AddNumberToObject(json, "math_flops", r->math_flops);

    cJSON *timings_array = cJSON_CreateDoubleArray(r->timings, iterationsPerConfig);
    cJSON_AddItemToObject(json, "timings", timings_array);

    return json;
}

cJSON* measurement_to_json(struct measurement *m, int numInputSizes, int iterationsPerConfig) {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddItemToObject(root, "meta", measurement_configuration_to_json(&m->configuration));

    cJSON *runs_array = cJSON_CreateArray();
    for (int i = 0; i < numInputSizes; i++) {  // Assuming there is a runCount field or similar
        cJSON_AddItemToArray(runs_array, run_to_json(&m->runs[i], iterationsPerConfig));
    }
    cJSON_AddItemToObject(root, "runs", runs_array);

    return root;
}

void write_measurement_to_file(struct measurement *m, const char *path, const char *name, int numInputSizes, int iterationsPerConfig) {
    char buffer[512];
    snprintf(buffer, sizeof(buffer), "%s/%s.json", path, name);
    cJSON *json = measurement_to_json(m, numInputSizes, iterationsPerConfig);
    char *string = cJSON_Print(json);
    FILE *file = fopen(buffer, "w");
    if (file == NULL) {
        fprintf(stderr, "Failed to open the file for writing.\n");
        return;
    }
    fprintf(file, "%s", string);
    fclose(file);

    cJSON_Delete(json);
    free(string);
}

// Taken from https://stackoverflow.com/questions/53708076/what-is-the-proper-way-to-use-clock-gettime
void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td)
{
    td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
    td->tv_sec  = t2.tv_sec - t1.tv_sec;
    if (td->tv_sec > 0 && td->tv_nsec < 0)
    {
        td->tv_nsec += NS_PER_SECOND;
        td->tv_sec--;
    }
    else if (td->tv_sec < 0 && td->tv_nsec > 0)
    {
        td->tv_nsec -= NS_PER_SECOND;
        td->tv_sec++;
    }
}

template<class T>
flop_counts timeRun(double *timings, int iterations, int M, int K, int N, MatMul<T> func, LCG rng)
{
    // A * B = C
    // A is m*k (m rows, k columns)
    // B is k*n (k rows, n columns)
    // C is m*n (m rows, n columns)
    T* A = (T *) malloc(M * K * sizeof(T));
    gen_urand<T>(&rng, A, M * K);
    T* B = (T *) malloc(K * N * sizeof(T));
    gen_urand<T>(&rng, B, K * N);
    T* C = (T *) malloc(M * N * sizeof(T));
    flop_counts counts;


    for(int i = 0; i < iterations; i++)
    {
        struct timespec start, end, delta;
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        counts = func(A, B, C, M, K, N);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        sub_timespec(start, end, &delta);
        timings[i] = (double) delta.tv_sec*NS_PER_SECOND + delta.tv_nsec;
    }

    free(A);
    free(B);
    free(C);

    return counts;
}

template flop_counts timeRun<float>(double *timings, int iterations, int M, int K, int N, MatMul<float> func, LCG rng);
template flop_counts timeRun<double>(double *timings, int iterations, int M, int K, int N, MatMul<double> func, LCG rng);

template<class T>
void timeFunction(matmul_variant<T> *function, char *path, LCG rng) {
    printf("Time %s\n", function->name);
    // information set by makefile?:
    // flags, compiler, cpu model
    int powerOfMaxSize = 12;
    int powerOfMinSize = 8;
    int numSizes = powerOfMaxSize - powerOfMinSize + 1;
    const int iterationsPerConfig = 10;

    struct measurementConfiguration runConfig = {
        .cpuModel = CPU,
        .gpuModel = GPU,
        .targetFunction = function->name,
    };

    double *timings = (double*)calloc(numSizes * iterationsPerConfig, sizeof(*timings));
    struct run *runs = (struct run*)calloc(numSizes, sizeof(*runs));
    for (int i = 0; i < numSizes; i++)
    {
        int n = 1 << (i + powerOfMinSize);
        flop_counts counts = timeRun<T>(&timings[i * iterationsPerConfig], iterationsPerConfig, n, n, n, function->function, rng);
        struct run run = {
            .M = n,
            .N = n,
            .K = n,
            .flops16 = counts.flops16,
            .flops32 = counts.flops32,
            .flops64 = counts.flops64,
            .math_flops = 2L*n*n*n,
            .timings = &timings[i * iterationsPerConfig]
        };
        runs[i] = run;
    }
    struct measurement measurement = {
        .configuration = runConfig,
        .runs = runs
    };
    write_measurement_to_file(&measurement, path, function->name, numSizes, iterationsPerConfig);
}

template void timeFunction<float>(matmul_variant<float> *function, char *path, LCG rng);
template void timeFunction<double>(matmul_variant<double> *function, char *path, LCG rng);

flop_counts matmul_flopcount_32(int M, int K, int N) {
    flop_counts counts;
    counts.flops16 = 0;
    counts.flops32 = 2L*M*K*N;
    counts.flops64 = 0;
    return counts;
}

flop_counts matmul_flopcount_64(int M, int K, int N) {
    flop_counts counts;
    counts.flops16 = 0;
    counts.flops32 = 2L*M*K*N;
    counts.flops64 = 0;
    return counts;
}
