#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include "machine.h"
#include "../lib/cjson/cJSON.h"
#include "precision.h"
#include "matcache.h"


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

cJSON* prec_measurement_to_json(struct precisionMeasurement *m, int precisionIterations) {
    cJSON *m_json = cJSON_CreateObject();
    if (!m_json) return NULL;

    cJSON_AddNumberToObject(m_json, "input_type", m->input_type);

    cJSON *residuals_array = cJSON_CreateDoubleArray(m->residuals, precisionIterations);
    cJSON_AddItemToObject(m_json, "residuals", residuals_array);

    return m_json;
}

cJSON* run_to_json(struct run *r, int iterations, int numInputTypes, int precisionIterationsPerInputType) {
    cJSON *json = cJSON_CreateObject();
    if (!json) return NULL;

    cJSON_AddNumberToObject(json, "M", r->M);
    cJSON_AddNumberToObject(json, "K", r->K);
    cJSON_AddNumberToObject(json, "N", r->N);
    cJSON_AddNumberToObject(json, "flops16", r->flops16);
    cJSON_AddNumberToObject(json, "flops32", r->flops32);
    cJSON_AddNumberToObject(json, "flops64", r->flops64);
    cJSON_AddNumberToObject(json, "math_flops", r->math_flops);

    cJSON *timings_array = cJSON_CreateDoubleArray(r->timings, iterations);
    cJSON_AddItemToObject(json, "timings", timings_array);

    if (precisionIterationsPerInputType > 0) {
        cJSON *precMs_array = cJSON_CreateArray();
        for (int i = 0; i < numInputTypes; i++) {
            cJSON_AddItemToArray(precMs_array, prec_measurement_to_json(&r->precMs[i], precisionIterationsPerInputType));
        }
        cJSON_AddItemToObject(json, "precMs", precMs_array);
    }

    return json;
}

cJSON* measurement_to_json(struct measurement *m, int numInputSizes, int numInputTypes, int *iterationsPerConfig, int *precisionIterationsPerInputType) {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddItemToObject(root, "meta", measurement_configuration_to_json(&m->configuration));

    cJSON *runs_array = cJSON_CreateArray();
    for (int i = 0; i < numInputSizes; i++) {  // Assuming there is a runCount field or similar
        cJSON_AddItemToArray(runs_array, run_to_json(&m->runs[i], iterationsPerConfig[i], numInputTypes, precisionIterationsPerInputType[i]));
    }
    cJSON_AddItemToObject(root, "runs", runs_array);

    return root;
}

void write_measurement_to_file(struct measurement *m, const char *path, const char *name, int numInputSizes, int numInputTypes, int *iterationsPerConfig, int *precisionIterationsPerInputType) {
    char buffer[512];
    snprintf(buffer, sizeof(buffer), "%s/%s.json", path, name);
    cJSON *json = measurement_to_json(m, numInputSizes, numInputTypes, iterationsPerConfig, precisionIterationsPerInputType);
    char *string = cJSON_Print(json);
    FILE *file = fopen(buffer, "w");
    if (file == NULL) {
        fprintf(stderr, "\rFailed to open the file for writing.\n");
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
flop_counts timeRun(double *timings, int iterations, int warmupIterations, size_t M, size_t K, size_t N, MatMul<T> func, LCG rng)
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

    for(int i = 0; i < warmupIterations; i++) {
        printf("\r\tWarmup Iteration %d/%d | Matrix Sizes: M=%zd, K=%zd, N=%zd", i + 1, warmupIterations, M, K, N);
        fflush(stdout);
        counts = func(A, B, C, M, K, N);
    }

    printf("\r%*s\r", 100, ""); // clear warmup progress line

    for(int i = 0; i < iterations; i++)
    {
        printf("\r\tTiming Iteration %d/%d | Matrix Sizes: M=%zd, K=%zd, N=%zd", i + 1, iterations, M, K, N);
        fflush(stdout);
        struct timespec start, end, delta;
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        counts = func(A, B, C, M, K, N);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        sub_timespec(start, end, &delta);
        timings[i] = (double) delta.tv_sec*NS_PER_SECOND + delta.tv_nsec;
    }

    printf("\r%*s\r", 100, ""); // clear iteration progress line

    free(A);
    free(B);
    free(C);

    return counts;
}

template flop_counts timeRun<float>(double *timings, int iterations, int warmupIterations, size_t M, size_t K, size_t N, MatMul<float> func, LCG rng);
template flop_counts timeRun<double>(double *timings, int iterations, int warmupIterations, size_t M, size_t K, size_t N, MatMul<double> func, LCG rng);

template<class T>
void measurePrecision(int input_type, double *residuals, int iterations, size_t M, size_t K, size_t N, MatMul<T> func, LCG rng)
{
    if (iterations == 0)
        return;
    
    // A * B = C
    // A is m*k (m rows, k columns)
    // B is k*n (k rows, n columns)
    // C is m*n (m rows, n columns)
    T* A = (T *) malloc(M * K * sizeof(T));
    T* B = (T *) malloc(K * N * sizeof(T));
    T* C = (T *) calloc(M * N, sizeof(T));
    T* C_ref = (T *) calloc(M * N, sizeof(T));

    for(int i = 0; i < iterations; i++) {
        printf("\r\tType %d Precision Measurement Iteration %d/%d | Matrix Sizes: M=%zd, K=%zd, N=%zd", input_type, i + 1, iterations, M, K, N);
        fflush(stdout);

        // Fill matrices A and B according to input type
        //fill_matrices<T>(&rng, input_type, A, B, M*K, K*N);
        auto [A, B, C_ref] = getMatrices<T>(M, K, N, input_type, &rng);

        // Run matmul implementation
        func(A, B, C, M, K, N);        

        // Measure error
        //referenceMatmul_full(A, B, C_ref, M, K, N);
        double residual = rel_residual(C, C_ref, M * N);
        residuals[i] = residual;
    }

    printf("\r%*s\r", 100, ""); // clear iteration progress line
    free(A);
    free(B);
    free(C);
    free(C_ref);
}


template void measurePrecision<float>(int input_type, double *residuals, int iterations, size_t M, size_t K, size_t N, MatMul<float> func, LCG rng);
template void measurePrecision<double>(int input_type, double *residuals, int iterations, size_t M, size_t K, size_t N, MatMul<double> func, LCG rng);

template<class T>
void timeFunction(matmul_variant<T> *function, char *path, LCG rng) {
    printf("Benchmark %s\n", function->name);
    // information set by makefile?:
    // flags, compiler, cpu model
    int powerOfMaxSize = 13;
    int powerOfMinSize = 7;
    int numSizes = powerOfMaxSize - powerOfMinSize + 1;
    const int numInputTypes = 5;
    const int maxIterationsPerConfig = 50;
    const int maxIterationsPerInputType = maxIterationsPerConfig / numInputTypes;
    int *iterationsPerConfig = (int*) calloc(numSizes, sizeof(*iterationsPerConfig));
    int *precisionIterationsPerInputType = (int*) calloc(numSizes, sizeof(*precisionIterationsPerInputType));
    const int warmupIterations = 1;

    struct measurementConfiguration runConfig = {
        .cpuModel = CPU,
        .gpuModel = GPU,
        .targetFunction = function->name,
    };

    double *timings = (double*) calloc(numSizes * maxIterationsPerConfig, sizeof(*timings));
    double *residuals = (double*) calloc(numSizes * numInputTypes * maxIterationsPerInputType, sizeof(*residuals));
    struct run *runs = (struct run*)calloc(numSizes, sizeof(*runs));
    struct precisionMeasurement *ms = (struct precisionMeasurement*) calloc(numSizes * numInputTypes, sizeof(*ms));
    for (int i = 0; i < numSizes; i++)
    {
        size_t n = 1 << (i + powerOfMinSize);

        // Adapt number of iterations based on heuristics on runtime
        iterationsPerConfig[i] = maxIterationsPerConfig;
        precisionIterationsPerInputType[i] = maxIterationsPerInputType;

        if (n >= 1 << 13) {
            iterationsPerConfig[i] = 1;
        } else if (n >= 1 << 12) {
            iterationsPerConfig[i] = 5;
        }
        
        if (n >= 1 << 11) {
            precisionIterationsPerInputType[i] = 0;
        } else if (n >= 1 << 10) {
            precisionIterationsPerInputType[i] = 1;
        } else if (n >= 1 << 9) {
            precisionIterationsPerInputType[i] = 5;
        }

        flop_counts counts = timeRun<T>(&timings[i * maxIterationsPerConfig], iterationsPerConfig[i], warmupIterations, n, n, n, function->function, rng);
        
        int m_idx = i * numInputTypes;
        for (int input_type = 0; input_type < numInputTypes; input_type++) {
            double *residuals_idx = &residuals[(i * numInputTypes + input_type) * maxIterationsPerInputType];
            measurePrecision<T>(input_type, residuals_idx, precisionIterationsPerInputType[i], n, n, n, function->function, rng);
            struct precisionMeasurement m = {
                .input_type = input_type,
                .residuals = residuals_idx,
            };
            ms[m_idx + input_type] = m;
        }

        struct run run = {
            .M = n,
            .N = n,
            .K = n,
            .flops16 = counts.flops16,
            .flops32 = counts.flops32,
            .flops64 = counts.flops64,
            .math_flops = 2L*n*n*n,
            .timings = &timings[i * maxIterationsPerConfig],
            .precMs = &ms[m_idx],
        };
        runs[i] = run;
    }
    struct measurement measurement = {
        .configuration = runConfig,
        .runs = runs
    };
    write_measurement_to_file(&measurement, path, function->name, numSizes, numInputTypes, iterationsPerConfig, precisionIterationsPerInputType);

    free(iterationsPerConfig);
    free(precisionIterationsPerInputType);
    free(ms);
    free(runs);
}

template void timeFunction<float>(matmul_variant<float> *function, char *path, LCG rng);
template void timeFunction<double>(matmul_variant<double> *function, char *path, LCG rng);

flop_counts matmul_flopcount_32(size_t M, size_t K, size_t N) {
    flop_counts counts;
    counts.flops16 = 0;
    counts.flops32 = 2L*M*K*N;
    counts.flops64 = 0;
    return counts;
}

flop_counts matmul_flopcount_64(size_t M, size_t K, size_t N) {
    flop_counts counts;
    counts.flops16 = 0;
    counts.flops32 = 2L*M*K*N;
    counts.flops64 = 0;
    return counts;
}
