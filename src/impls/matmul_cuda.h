#ifndef MATMUL_CUDA_H
#define MATMUL_CUDA_H

#include <cuda_fp16.h>

//NOTE(max): this is constant for now, if we have architectures that have
//different warp sizes we need to make this dynamic
#define WARP_SIZE 32

typedef struct alignas(8) 
{
    half x, y, z, w;
} half4;

struct matmulTemplateArgsCUDA
{
    int BM; // The number of rows of C a threadblock computes
    int BN; // The number of cols of C a threadblock computes
    int BK; // The dimension of the "dotproducts" a threadblock performs in each iteration 
    int WM; // The number of rows of C a warp computes
    int WN; // The number of cols of C a warp computes
    int TM; // The number of rows of C a thread computes
    int TN; // The number of cols of C a thread computes
    int WMITER; // The number of warpTiles in the M dimension of C
    int WNITER; // The number of warpTiles in the N dimension of C
    int threadsPerBlock; // The amount of threads a threadblock needs
};

struct matmulScalesCUDA
{
    int scaleBM;
    int scaleBN;
    int scaleBK;
    int scaleWM;
    int scaleWN;
    int scaleTM;
    int scaleTN;
};

struct matmulTemplateArgsTensor
{
    int BM; // The number of rows of C a threadblock computes
    int BN; // The number of cols of C a threadblock computes
    int BK; // The dimension of the "dotproducts" a threadblock performs in each iteration 
    int WM; // The number of rows of C a warp computes
    int WN; // The number of cols of C a warp computes
    int CHUNK_K; // BK / WMMA_K
    int N_WARP_ROWS_PER_BLOCK; // How many rows of warps a threadblock gets assigned
    int N_WARP_COLS_PER_BLOCK; // How many cols of warps a threadblock gets assigned
    int N_WMMA_ROWS_PER_WARP; // The amount of tensor core multiplications required to cover WM
    int N_WMMA_COLS_PER_WARP; // The amount of tensor core multiplications required to cover WN
    int threadsPerBlock; // The amount of threads a threadblock needs
};

struct matmulScalesTensor
{
    int scaleBM;
    int scaleBN;
    int scaleChunk;
    int scaleWM;
    int scaleWN;
};


template<typename InputType, typename OutputType, int version>
void matmulTensorCores(InputType *A, InputType *B, OutputType *C, size_t M, size_t K, size_t N);

template<typename InputType, typename OutputType, int version>
void matmulCUDACores(InputType *A, InputType *B, OutputType *C, size_t M, size_t K, size_t N);

#endif // MATMUL_CUDA_H
