#ifndef MATMUL_H
#define MATMUL_H

#include <stdio.h>
#include <stdlib.h>

void matmul_v0(double *A, double *B, double *C, int M, int K, int N);
// void matmul_v1(double *A, double *B, double *C, int M, int K, int N);
// void matmul_v2(double *A, double *B, double *C, int M, int K, int N);
// void matmul_v3(double *A, double *B, double *C, int M, int K, int N);

void matmul_cuda_v0(double *A, double *B, double *C, int M, int K, int N);
void matmul_simpleMarkidis_v0(float *A, float *B, float *C, int M, int K, int N);
void matmul_simpleMarkidis_v1(float *A, float *B, float *C, int M, int K, int N);
void matmul_simpleMarkidis_v2(float *A, float *B, float *C, int M, int K, int N);
void matmul_simpleMarkidis_v3(float *A, float *B, float *C, int M, int K, int N);
void matmul_simpleMarkidis_v4(float *A, float *B, float *C, int M, int K, int N);
void matmul_simpleOotomo_v0(float *A, float *B, float *C, int M, int K, int N);

void matmul_Oootomo_v0(float *A, float *B, float *C, int M, int K, int N);
void matmul_Oootomo_v1(float *A, float *B, float *C, int M, int K, int N);

void matmul_cuBLAS32(float *A, float *B, float *C, int M, int K, int N);
void matmul_cuBLAS64(double *A, double *B, double *C, int M, int K, int N);

#endif // MATMUL_H
