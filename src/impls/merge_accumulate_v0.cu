#include <cuda_fp16.h>

#include "../merge_accumulate.h"

void merge_v0(const void *A16, const void *dA16, double* merged, size_t M, size_t N)
{
    const half *_A16 = (half *) A16;
    const half *_dA16 = (half *) dA16;
    for (int i = 0; i < M * N; ++i)
        merged[i] = (double) __half2float(_A16[i]) + (double) __half2float(_dA16[i]);
}

void mergef_v0(const void *A16, const void *dA16, float* merged, size_t M, size_t N)
{
    const half *_A16 = (half *) A16;
    const half *_dA16 = (half *) dA16;
    for (int i = 0; i < M * N; ++i)
        merged[i] = __half2float(_A16[i]) + __half2float(_dA16[i]);
}

void merge_Ootomo_v0(const void *A16, const void *dA16, double* merged, size_t M, size_t N)
{
    const half *_A16 = (half *) A16;
    const half *_dA16 = (half *) dA16;
    for (int i = 0; i < M * N; ++i)
        merged[i] = (double) __half2float(_A16[i]) + (double) (__half2float(_dA16[i])) / 2048.0; // 2^11
}

void mergef_Ootomo_v0(const void *A16, const void *dA16, float* merged, size_t M, size_t N)
{
    const half *_A16 = (half *) A16;
    const half *_dA16 = (half *) dA16;
    for (int i = 0; i < M * N; ++i)
        merged[i] = __half2float(_A16[i]) + (__half2float(_dA16[i])) / 2048.0; // 2^11
}

void accumulate_Ootomo_v0(
    const float *A16B16,
    const float *dA16B16,
    const float *A16dB16,
    float *C,
    const int rows,
    const int cols
)
{
    for (int i = 0; i < rows * cols; ++i)
        C[i] = A16B16[i] + (dA16B16[i] + A16dB16[i]) / 2048.f; // 2^11
}

