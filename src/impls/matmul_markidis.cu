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

#define WARP_SIZE 32

//blockDim.x == warpSize
//blockDim.y == BlockSizeN / (WarpSizeN * FragSizeN)
//blockDim.z == BlockSizeM / (WarpSizeM * FragSizeM)
//gridDim.x == RoundUp(N / BlockSizeN)
//gridDim.y == RoundUp(M / BlockSizeM)
//KStep % warpSize == 0
template<int BlockSizeM, int BlockSizeN, int KStep, 
         int WarpSizeM, int WarpSizeN,
         int FragSizeM, int FragSizeK, int FragSizeN>
__global__ void matmul_markidis_kernel(float *A, float *B, float *C, int M, int K, int N)
{
    using namespace nvcuda;

    __shared__ half SharedA[2][BlockSizeM][KStep];
    __shared__ half SharedB[2][KStep][BlockSizeN];

    const int scalar_blockMBase = blockIdx.y * BlockSizeM;
    const int scalar_blockNBase = blockIdx.x * BlockSizeN;
    const int scalar_blockBaseA = scalar_blockMBase * K;
    const int scalar_blockBaseB = scalar_blockNBase;
    const int scalar_blockBaseC = scalar_blockMBase * N + scalar_blockNBase;

    const int warpNOffset = threadIdx.y * (WarpSizeN * FragSizeN);
    const int warpMOffset = threadIdx.z * (WarpSizeM * FragSizeM);
    const int threadIndex = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    const int loadARowStep = (blockDim.x * blockDim.y * blockDim.z) / KStep;
    const int loadAMOffset = threadIndex / KStep;
    const int loadAKOffset = threadIndex % KStep;

    const int loadBRowStep = (blockDim.x * blockDim.y * blockDim.z) / BlockSizeN;
    const int loadBKOffset = threadIndex / BlockSizeN;
    const int loadBNOffset = threadIndex % BlockSizeN;

    wmma::fragment<wmma::matrix_a, FragSizeM, FragSizeN, FragSizeK, half, wmma::row_major> aFrag[2][WarpSizeM];
    wmma::fragment<wmma::matrix_b, FragSizeM, FragSizeN, FragSizeK, half, wmma::row_major> bFrag[2][WarpSizeN];
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
            int m = mOffset + loadAMOffset;
            float original = A[loadABase + m * K + loadAKOffset];
            half a0 = (half)original;
            half a1 = (half)(original - (float)a0);
            SharedA[0][m][loadAKOffset] = a0;
            SharedA[1][m][loadAKOffset] = a1;
        }
        for(int kOffset = 0; kOffset < KStep; kOffset += loadBRowStep)
        {
            int k = kOffset + loadBKOffset;
            float original = B[loadBBase + k * N + loadBNOffset];
            half b0 = (half)original;
            half b1 = (half)(original - (float)b0);
            SharedB[0][k][loadBNOffset] = b0;
            SharedB[1][k][loadBNOffset] = b1;
        }

        __syncthreads();

        for(int kOffset = 0; kOffset < KStep; kOffset += FragSizeK)
        {
            for(int m = 0; m < WarpSizeM; m++)
            {
                int mIndex = warpMOffset + m * FragSizeM;
                wmma::load_matrix_sync(aFrag[0][m], &SharedA[0][mIndex][kOffset], KStep);
                wmma::load_matrix_sync(aFrag[1][m], &SharedA[1][mIndex][kOffset], KStep);
            }
            for(int n = 0; n < WarpSizeN; n++)
            {
                int nIndex = warpNOffset + n * FragSizeN;
                wmma::load_matrix_sync(bFrag[0][n], &SharedB[0][kOffset][nIndex], BlockSizeN);
                wmma::load_matrix_sync(bFrag[1][n], &SharedB[1][kOffset][nIndex], BlockSizeN);
            }
            for(int m = 0; m < WarpSizeM; m++)
                for(int n = 0; n < WarpSizeN; n++)
                    for(int i = 0; i < 4; i++)
                        wmma::mma_sync(cFrag[m][n], aFrag[i/2][m], bFrag[i%2][n], cFrag[m][n]);
        }

        __syncthreads();
    }

    int offsetC = scalar_blockBaseC + warpMOffset * N + warpNOffset;
    for(int m = 0; m < WarpSizeM; m++)
        for(int n = 0; n < WarpSizeN; n++)
            wmma::store_matrix_sync(C + offsetC + (m * FragSizeM * N + n * FragSizeN) , cFrag[m][n], N, wmma::mem_row_major);
}

flop_counts matmul_markidis(float *A, float *B, float *C, int M, int K, int N) 
{
    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_N = 64;
    const int K_STEP       = 32;
    const int WARP_SIZE_M  = 2;
    const int WARP_SIZE_N  = 2;
    const int FRAG_SIZE_M  = 16;
    const int FRAG_SIZE_K  = 16;
    const int FRAG_SIZE_N  = 16;

    assert((M & 0xF) == 0);
    assert((K & 0xF) == 0);
    assert((N & 0xF) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate");
    size_t ASize = M * K * sizeof(float);
    size_t BSize = K * N * sizeof(float);
    size_t CSize = M * N * sizeof(float);

    float *deviceA, *deviceB, *deviceC;
    PRINT_ON_ERROR(cudaMalloc(&deviceA, ASize));
    PRINT_ON_ERROR(cudaMalloc(&deviceB, BSize));
    PRINT_ON_ERROR(cudaMalloc(&deviceC, CSize));

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    PRINT_ON_ERROR(cudaMemcpy(deviceA, A, ASize, cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(deviceB, B, BSize, cudaMemcpyHostToDevice));

    PROFILE_SEGMENTS_SWITCH("matmul");
    dim3 threadsPerBlock(WARP_SIZE, 
                         BLOCK_SIZE_N / (WARP_SIZE_N * FRAG_SIZE_N), 
                         BLOCK_SIZE_M / (WARP_SIZE_M * FRAG_SIZE_M));
    dim3 blocks(DivRoundUp(N, BLOCK_SIZE_N), DivRoundUp(M, BLOCK_SIZE_M));
    matmul_markidis_kernel<
              BLOCK_SIZE_M, BLOCK_SIZE_N, K_STEP,
              WARP_SIZE_M, WARP_SIZE_N,
              FRAG_SIZE_M, FRAG_SIZE_K, FRAG_SIZE_N>
             <<<blocks, threadsPerBlock>>>
             (deviceA, deviceB, deviceC, M, K, N);
    PRINT_ON_ERROR(cudaGetLastError());

    PRINT_ON_ERROR(cudaDeviceSynchronize());

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    PRINT_ON_ERROR(cudaMemcpy(C, deviceC, CSize, cudaMemcpyDeviceToHost));

    PROFILE_SEGMENTS_SWITCH("free");
    PRINT_ON_ERROR(cudaFree(deviceA));
    PRINT_ON_ERROR(cudaFree(deviceB));
    PRINT_ON_ERROR(cudaFree(deviceC));
    PROFILE_SEGMENT_FUNCTION_END();

    flop_counts counts = {8L*M*K*N, 0, 0};
    return counts;
}

