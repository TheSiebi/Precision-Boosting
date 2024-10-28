#include "timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include "machine.h"
#include "../lib/cjson/cJSON.h"

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



void timeRun(double *timings, int iterations, int M, int K, int N, MatMul func)
{
    // A * B = C
    // A is m*k (m rows, k columns)
    // B is k*n (k rows, n columns)
    // C is m*n (m rows, n columns)
    double* A = (double *) malloc(M * K * sizeof(double));
    double* B = (double *) malloc(K * N * sizeof(double));
    double* C = (double *) malloc(M * N * sizeof(double));

    for(int i = 0; i < iterations; i++)
    {
        // Populate matrices with random values between 0 and 1
        for (int j = 0; j < M*K; j++) {
            A[j] = (double) rand() / (double) RAND_MAX;
        }
        for (int j = 0; j < K*N; j++) {
            B[j] = (double) rand() / (double) RAND_MAX;
        }

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        func(A, B, C, M, K, N);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        timings[i] = (double) (end.tv_nsec - start.tv_nsec);
    }

    free(A);
    free(B);
    free(C);
}


void timeFunction(struct matmul_variant *function, char *path) {
    printf("Time %s\n", function->name);
    // information set by makefile?:
    // flags, compiler, cpu model
    int powerOfMaxSize = 8;
    int powerOfMinSize = 3;
    int numSizes = powerOfMaxSize - powerOfMinSize + 1;
    const int iterationsPerConfig = 5;

    struct measurementConfiguration runConfig = {
        .targetFunction = function->name,
        .cpuModel = CPU,
        .gpuModel = GPU
    };

    double *timings = calloc(numSizes * iterationsPerConfig, sizeof(*timings));
    struct run *runs = calloc(numSizes, sizeof(*runs));
    for (int i = 0; i < numSizes; i++)
    {
        int n = 1 << (i + powerOfMinSize);
        timeRun(&timings[i * iterationsPerConfig], iterationsPerConfig, n, n, n, function->function);
        struct run run = {
            .M = n,
            .N = n,
            .K = n,
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
