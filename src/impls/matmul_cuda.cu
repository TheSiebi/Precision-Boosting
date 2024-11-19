#include <assert.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <driver_types.h>

#include "../matmul.h"
#include "../profiler.h"
#include "../cuda_utils.h"
#include "../timer.h"

__global__ void matmul_cuda_v0_kernel(double *A, double *B, double *C, int M, int K, int N) 
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    double result = 0.0;
    for (int k = 0; k < K; k++) 
    {
        result += A[m*K + k] * B[k*N + n];
    }
    C[m*N + n] = result;
}

__global__ void matmul_kernel_v0(half *A, half *B, float *C, int M, int K, int N) 
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    float result = 0.0;
    for (int k = 0; k < K; k++) 
    {
        result += (float)A[m*K + k] * (float)B[k*N + n];
    }
    C[m*N + n] = result;
}

__global__ void matmul_kernel_v1(half *A, half *B, float *C, int M, int K, int N) 
{
    using namespace nvcuda;

    int warpM = blockIdx.x * 16;
    int warpN = blockIdx.y * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;

    wmma::fill_fragment(cFrag, 0);

    for (int k = 0; k < K; k += 16) 
    {
        wmma::load_matrix_sync(aFrag, A + warpM * K + k, K);
        wmma::load_matrix_sync(bFrag, B + k * N + warpN, N);

        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    wmma::store_matrix_sync(C + warpM * N + warpN, cFrag, N, wmma::mem_row_major);
}
//NOTE(max): this is constant for now, if we have architectures that have
//different warp sizes we need to make this dynamic
#define WARP_SIZE 32

//blockDim.x == warpSize
//blockDim.y == BlockSizeN / FragSizeN
//blockDim.z == BlockSizeM / FragSizeM
//gridDim.x == RoundUp(N / BlockSizeN)
//gridDim.y == RoundUp(M / BlockSizeM)
template<int BlockSizeM, int BlockSizeN, int KStep, 
         int FragSizeM, int FragSizeK, int FragSizeN>
__global__ void matmul_kernel_v2(half *A, half *B, float *C, int M, int K, int N)
{
    using namespace nvcuda;

    const int scalar_blockMBase = blockIdx.y * BlockSizeM;
    const int scalar_blockNBase = blockIdx.x * BlockSizeN;
    const int scalar_blockBaseA = scalar_blockMBase * K;
    const int scalar_blockBaseB = scalar_blockNBase;
    const int scalar_blockBaseC = scalar_blockMBase * N + scalar_blockNBase;

    const int warpNOffset = threadIdx.y * FragSizeN;
    const int warpMOffset = threadIdx.z * FragSizeM;

    wmma::fragment<wmma::matrix_a, FragSizeM, FragSizeN, FragSizeK, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, FragSizeM, FragSizeN, FragSizeK, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, FragSizeM, FragSizeN, FragSizeK, float> cFrag;

    wmma::fill_fragment(cFrag, 0);

    for (int kBase = 0; kBase < K; kBase += KStep) 
    {
        for(int kOffset = 0; kOffset < KStep; kOffset += FragSizeK)
        {
            int k = kBase + kOffset;
            int offsetA = scalar_blockBaseA + warpMOffset * K + k;
            int offsetB = scalar_blockBaseB + k * N + warpNOffset;
            wmma::load_matrix_sync(aFrag, A + offsetA, K);
            wmma::load_matrix_sync(bFrag, B + offsetB, N);

            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

    int offsetC = scalar_blockBaseC + warpMOffset * N + warpNOffset;
    wmma::store_matrix_sync(C + offsetC, cFrag, N, wmma::mem_row_major);
}


//blockDim.x == warpSize
//blockDim.y == BlockSizeN / FragSizeN
//blockDim.z == BlockSizeM / FragSizeM
//gridDim.x == RoundUp(N / BlockSizeN)
//gridDim.y == RoundUp(M / BlockSizeM)
//KStep % warpSize == 0
template<int BlockSizeM, int BlockSizeN, int KStep, 
         int FragSizeM, int FragSizeK, int FragSizeN>
__global__ void matmul_kernel_v3(half *A, half *B, float *C, int M, int K, int N)
{
    using namespace nvcuda;

    __shared__ half SharedA[BlockSizeM][KStep];
    __shared__ half SharedB[KStep][BlockSizeN];

    const int scalar_blockMBase = blockIdx.y * BlockSizeM;
    const int scalar_blockNBase = blockIdx.x * BlockSizeN;
    const int scalar_blockBaseA = scalar_blockMBase * K;
    const int scalar_blockBaseB = scalar_blockNBase;
    const int scalar_blockBaseC = scalar_blockMBase * N + scalar_blockNBase;

    const int warpNOffset = threadIdx.y * FragSizeN;
    const int warpMOffset = threadIdx.z * FragSizeM;
    const int warpIndex = threadIdx.y + blockDim.y * threadIdx.z;

    constexpr int LoadAWarpsPerRow = KStep / WARP_SIZE;
    const int loadARowStep = (blockDim.y * blockDim.z) / LoadAWarpsPerRow;
    const int loadAWarpMOffset = warpIndex / LoadAWarpsPerRow;
    const int loadAThreadOffset = (warpIndex % LoadAWarpsPerRow) * WARP_SIZE + threadIdx.x;

    constexpr int LoadBWarpsPerRow = BlockSizeN / WARP_SIZE;
    const int loadBRowStep = (blockDim.y * blockDim.z) / LoadBWarpsPerRow;
    const int loadBWarpKOffset = warpIndex / LoadBWarpsPerRow;
    const int loadBThreadOffset = (warpIndex % LoadBWarpsPerRow) * WARP_SIZE + threadIdx.x;

    wmma::fragment<wmma::matrix_a, FragSizeM, FragSizeN, FragSizeK, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, FragSizeM, FragSizeN, FragSizeK, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, FragSizeM, FragSizeN, FragSizeK, float> cFrag;

    wmma::fill_fragment(cFrag, 0);

    for (int kBase = 0; kBase < K; kBase += KStep) 
    {
        const int loadABase = scalar_blockBaseA + kBase;
        const int loadBBase = scalar_blockBaseB + kBase * N;
        for(int mOffset = 0; mOffset < BlockSizeM; mOffset += loadARowStep)
        {
            int m = mOffset + loadAWarpMOffset;
            SharedA[m][loadAThreadOffset] = A[loadABase + m * K + loadAThreadOffset];
        }
        for(int kOffset = 0; kOffset < KStep; kOffset += loadBRowStep)
        {
            int k = kOffset + loadBWarpKOffset;
            SharedB[k][loadBThreadOffset] = B[loadBBase + k * N + loadBThreadOffset];
        }

        __syncthreads();

        for(int kOffset = 0; kOffset < KStep; kOffset += FragSizeK)
        {
            wmma::load_matrix_sync(aFrag, &SharedA[warpMOffset][kOffset], KStep);
            wmma::load_matrix_sync(bFrag, &SharedB[kOffset][warpNOffset], BlockSizeN);

            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }

        __syncthreads();
    }

    int offsetC = scalar_blockBaseC + warpMOffset * N + warpNOffset;
    wmma::store_matrix_sync(C + offsetC, cFrag, N, wmma::mem_row_major);
}

//blockDim.x == warpSize
//blockDim.y == BlockSizeN / (WarpSizeN * FragSizeN)
//blockDim.z == BlockSizeM / (WarpSizeM * FragSizeM)
//gridDim.x == RoundUp(N / BlockSizeN)
//gridDim.y == RoundUp(M / BlockSizeM)
//KStep % warpSize == 0
template<int BlockSizeM, int BlockSizeN, int KStep, 
         int WarpSizeM, int WarpSizeN,
         int FragSizeM, int FragSizeK, int FragSizeN>
__global__ void matmul_kernel_v4(half *A, half *B, float *C, int M, int K, int N)
{
    using namespace nvcuda;

    __shared__ half SharedA[BlockSizeM][KStep];
    __shared__ half SharedB[KStep][BlockSizeN];

    const int scalar_blockMBase = blockIdx.y * BlockSizeM;
    const int scalar_blockNBase = blockIdx.x * BlockSizeN;
    const int scalar_blockBaseA = scalar_blockMBase * K;
    const int scalar_blockBaseB = scalar_blockNBase;
    const int scalar_blockBaseC = scalar_blockMBase * N + scalar_blockNBase;

    const int warpNOffset = threadIdx.y * (WarpSizeN * FragSizeN);
    const int warpMOffset = threadIdx.z * (WarpSizeM * FragSizeM);
    const int warpIndex = threadIdx.y + blockDim.y * threadIdx.z;

    constexpr int LoadAWarpsPerRow = KStep / WARP_SIZE;
    const int loadARowStep = (blockDim.y * blockDim.z) / LoadAWarpsPerRow;
    const int loadAWarpMOffset = warpIndex / LoadAWarpsPerRow;
    const int loadAThreadOffset = (warpIndex % LoadAWarpsPerRow) * WARP_SIZE + threadIdx.x;

    constexpr int LoadBWarpsPerRow = BlockSizeN / WARP_SIZE;
    const int loadBRowStep = (blockDim.y * blockDim.z) / LoadBWarpsPerRow;
    const int loadBWarpKOffset = warpIndex / LoadBWarpsPerRow;
    const int loadBThreadOffset = (warpIndex % LoadBWarpsPerRow) * WARP_SIZE + threadIdx.x;

    wmma::fragment<wmma::matrix_a, FragSizeM, FragSizeN, FragSizeK, half, wmma::row_major> aFrag[WarpSizeM];
    wmma::fragment<wmma::matrix_b, FragSizeM, FragSizeN, FragSizeK, half, wmma::row_major> bFrag[WarpSizeN];
    wmma::fragment<wmma::accumulator, FragSizeM, FragSizeN, FragSizeK, float> cFrag[WarpSizeM][WarpSizeN];

    for(int m = 0; m < WarpSizeM; m++)
        for(int n = 0; n < WarpSizeN; n++)
            wmma::fill_fragment(cFrag[m][n], 0);

    for (int kBase = 0; kBase < K; kBase += KStep) 
    {
        const int loadABase = scalar_blockBaseA + kBase;
        const int loadBBase = scalar_blockBaseB + kBase * N;
        for(int mOffset = 0; mOffset < BlockSizeM; mOffset += loadARowStep)
        {
            int m = mOffset + loadAWarpMOffset;
            SharedA[m][loadAThreadOffset] = A[loadABase + m * K + loadAThreadOffset];
        }
        for(int kOffset = 0; kOffset < KStep; kOffset += loadBRowStep)
        {
            int k = kOffset + loadBWarpKOffset;
            SharedB[k][loadBThreadOffset] = B[loadBBase + k * N + loadBThreadOffset];
        }

        __syncthreads();

        for(int kOffset = 0; kOffset < KStep; kOffset += FragSizeK)
        {
            for(int m = 0; m < WarpSizeM; m++)
            {
                half *aOffset = &SharedA[warpMOffset + m * FragSizeM][kOffset];
                wmma::load_matrix_sync(aFrag[m], aOffset, KStep);
            }
            for(int n = 0; n < WarpSizeN; n++)
            {
                half *bOffset = &SharedB[kOffset][warpNOffset + n * FragSizeN];
                wmma::load_matrix_sync(bFrag[n], bOffset, BlockSizeN);
            }
            for(int m = 0; m < WarpSizeM; m++)
                for(int n = 0; n < WarpSizeN; n++)
                    wmma::mma_sync(cFrag[m][n], aFrag[m], bFrag[n], cFrag[m][n]);
        }

        __syncthreads();
    }

    int offsetC = scalar_blockBaseC + warpMOffset * N + warpNOffset;
    for(int m = 0; m < WarpSizeM; m++)
        for(int n = 0; n < WarpSizeN; n++)
            wmma::store_matrix_sync(C + offsetC + (m * FragSizeM * N + n * FragSizeN) , cFrag[m][n], N, wmma::mem_row_major);
}


template<typename InputType, typename OutputType, int version>
void matmul(InputType *A, InputType *B, OutputType *C, int M, int K, int N)
{
    if constexpr (version == 0)
    {
        dim3 threadsPerBlock(16, 16);
        dim3 blocks(M/threadsPerBlock.x, N/threadsPerBlock.y);
        matmul_kernel_v0<<<blocks, threadsPerBlock>>>(A, B, C, M, K, N);
        PRINT_ON_ERROR(cudaGetLastError());
    }
    else if constexpr (version == 1)
    {
        dim3 threadsPerBlock(32, 1);
        dim3 blocks(M/16, N/16);
        matmul_kernel_v1<<<blocks, threadsPerBlock>>>(A, B, C, M, K, N);
        PRINT_ON_ERROR(cudaGetLastError());
    }
    else if constexpr (version == 2 || version == 3)
    {
        const int BLOCK_SIZE_M = 64;
        const int BLOCK_SIZE_N = 64;
        const int K_STEP       = 32;
        const int FRAG_SIZE_M  = 16;
        const int FRAG_SIZE_K  = 16;
        const int FRAG_SIZE_N  = 16;
        dim3 threadsPerBlock(WARP_SIZE, BLOCK_SIZE_N / FRAG_SIZE_N, BLOCK_SIZE_M / FRAG_SIZE_M);
        dim3 blocks(DivRoundUp(N, BLOCK_SIZE_N), DivRoundUp(M, BLOCK_SIZE_M));
        if constexpr (version == 2)
        {
            matmul_kernel_v2<BLOCK_SIZE_M, BLOCK_SIZE_N, K_STEP,
                      FRAG_SIZE_M, FRAG_SIZE_K, FRAG_SIZE_N>
                     <<<blocks, threadsPerBlock>>>
                     (A, B, C, M, K, N);
        }
        else
        {
            matmul_kernel_v3<BLOCK_SIZE_M, BLOCK_SIZE_N, K_STEP,
                      FRAG_SIZE_M, FRAG_SIZE_K, FRAG_SIZE_N>
                     <<<blocks, threadsPerBlock>>>
                     (A, B, C, M, K, N);
        }
        PRINT_ON_ERROR(cudaGetLastError());
    }
    else if constexpr (version == 4)
    {
        const int BLOCK_SIZE_M = 128;
        const int BLOCK_SIZE_N = 128;
        const int K_STEP       = 32;
        const int WARP_SIZE_M  = 2;
        const int WARP_SIZE_N  = 2;
        const int FRAG_SIZE_M  = 16;
        const int FRAG_SIZE_K  = 16;
        const int FRAG_SIZE_N  = 16;
        dim3 threadsPerBlock(WARP_SIZE, 
                             BLOCK_SIZE_N / (WARP_SIZE_N * FRAG_SIZE_N), 
                             BLOCK_SIZE_M / (WARP_SIZE_M * FRAG_SIZE_M));
        dim3 blocks(DivRoundUp(N, BLOCK_SIZE_N), DivRoundUp(M, BLOCK_SIZE_M));
        matmul_kernel_v4<BLOCK_SIZE_M, BLOCK_SIZE_N, K_STEP,
                  WARP_SIZE_M, WARP_SIZE_N,
                  FRAG_SIZE_M, FRAG_SIZE_K, FRAG_SIZE_N>
                 <<<blocks, threadsPerBlock>>>
                 (A, B, C, M, K, N);
        PRINT_ON_ERROR(cudaGetLastError());
    }
}

template void matmul<half, float, 0>(half*, half*, float*, int, int, int);
template void matmul<half, float, 1>(half*, half*, float*, int, int, int);
template void matmul<half, float, 2>(half*, half*, float*, int, int, int);
template void matmul<half, float, 3>(half*, half*, float*, int, int, int);
template void matmul<half, float, 4>(half*, half*, float*, int, int, int);


flop_counts matmul_cuda_v0(double *A, double *B, double *C, int M, int K, int N) 
{
    assert((M & 0xF) == 0);
    assert((K & 0xF) == 0);
    assert((N & 0xF) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate");
    size_t ASize = M * K * sizeof(double);
    size_t BSize = K * N * sizeof(double);
    size_t CSize = M * N * sizeof(double);

    double *deviceA, *deviceB, *deviceC;
    PRINT_ON_ERROR(cudaMalloc(&deviceA, ASize));
    PRINT_ON_ERROR(cudaMalloc(&deviceB, BSize));
    PRINT_ON_ERROR(cudaMalloc(&deviceC, CSize));

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    PRINT_ON_ERROR(cudaMemcpy(deviceA, A, ASize, cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(deviceB, B, BSize, cudaMemcpyHostToDevice));

    PROFILE_SEGMENTS_SWITCH("matmul");
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(M/threadsPerBlock.x, N/threadsPerBlock.y);
    matmul_cuda_v0_kernel<<<blocks, threadsPerBlock>>>(deviceA, deviceB, deviceC, M, K, N);
    PRINT_ON_ERROR(cudaGetLastError());

    PRINT_ON_ERROR(cudaDeviceSynchronize());

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    PRINT_ON_ERROR(cudaMemcpy(C, deviceC, CSize, cudaMemcpyDeviceToHost));

    PROFILE_SEGMENTS_SWITCH("free");
    PRINT_ON_ERROR(cudaFree(deviceA));
    PRINT_ON_ERROR(cudaFree(deviceB));
    PRINT_ON_ERROR(cudaFree(deviceC));
    PROFILE_SEGMENT_FUNCTION_END();

    flop_counts counts = {0L, 0L, 2L*M*K*N};
    return counts;
}

