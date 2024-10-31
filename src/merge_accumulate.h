#ifndef MERGE_ACCUMULATE_H
#define MERGE_ACCUMULATE_H

void merge_v0(const void *A16, const void *dA16, double* merged, int M, int N);
void mergef_v0(const void *A16, const void *dA16, float* merged, int M, int N);
void merge_Ootomo_v0(const void *A16, const void *dA16, double* merged, int M, int N);
void mergef_Ootomo_v0(const void *A16, const void *dA16, float* merged, int M, int N);

// [4] Ootomo: equation 24, page 15
void accumulate_Ootomo_v0(
    const float *A16B16,
    const float *dA16B16,
    const float *A16dB16,
    float *C,
    const int rows,
    const int cols
);

#endif // MERGE_ACCUMULATE_H
