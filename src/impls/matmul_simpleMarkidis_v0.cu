#include <assert.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../cuda_utils.h"
#include "../matmul.h"
#include "../profiler.h"

__global__ void matmul_v0(half *A, half *B, float *C, int M, int K, int N) 
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

__global__ void matmul_v1(half *A, half *B, float *C, int M, int K, int N) 
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
__global__ void matmul_v2(half *A, half *B, float *C, int M, int K, int N)
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
__global__ void matmul_v3(half *A, half *B, float *C, int M, int K, int N)
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

__global__ void split_cuda(float *A, half *A0, half *A1, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < N)
    {
        float value = A[i];
        half mainPart = (half)value;
        A0[i] = mainPart;
        A1[i] = (half)(value - (float)mainPart);
    }
}

template<int version>
void matmul_simpleMarkidis(float *A, float *B, float *C, int M, int K, int N) 
{
    assert((M % 16) == 0);
    assert((K % 16) == 0);
    assert((N % 16) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate cpu");

    size_t ASize = M * K * sizeof(half);
    size_t BSize = K * N * sizeof(half);
    size_t CSize = M * N * sizeof(float);
    
    float *hostC[4];
    for(int i = 0; i < 4; i++)
        hostC[i] = (float*)malloc(CSize);


    PROFILE_SEGMENTS_SWITCH("allocate gpu");

    half *deviceA[2], *deviceB[2];
    float *deviceC[4];
    float *deviceAFull, *deviceBFull;
    for(int i = 0; i < 2; i++)
    {
        cudaGetLastError();
        PRINT_ON_ERROR(cudaMalloc(&deviceA[i], ASize));
        PRINT_ON_ERROR(cudaMalloc(&deviceB[i], BSize));
    }
    for(int i = 0; i < 4; i++)
        PRINT_ON_ERROR(cudaMalloc(&deviceC[i], CSize));
    PRINT_ON_ERROR(cudaMalloc(&deviceAFull, M*K*sizeof(float)));
    PRINT_ON_ERROR(cudaMalloc(&deviceBFull, K*N*sizeof(float)));

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");

    PRINT_ON_ERROR(cudaMemcpy(deviceAFull, A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(deviceBFull, B, K*N*sizeof(float), cudaMemcpyHostToDevice));

    PROFILE_SEGMENTS_SWITCH("split");

    split_cuda<<<DivRoundUp(M*K, 256), 256>>>(deviceAFull, deviceA[0], deviceA[1], M * K);
    PRINT_ON_ERROR(cudaGetLastError());
    split_cuda<<<DivRoundUp(K*N, 256), 256>>>(deviceBFull, deviceB[0], deviceB[1], K * N);
    PRINT_ON_ERROR(cudaGetLastError());

    PRINT_ON_ERROR(cudaDeviceSynchronize());

    PROFILE_SEGMENTS_SWITCH("matmul");
    if constexpr (version == 0)
    {
        dim3 threadsPerBlock(16, 16);
        dim3 blocks(M/threadsPerBlock.x, N/threadsPerBlock.y);
        for(int i = 0; i < 4; i++)
        {
            matmul_v0<<<blocks, threadsPerBlock>>>(deviceA[i/2], deviceB[i%2], deviceC[i], M, K, N);
            PRINT_ON_ERROR(cudaGetLastError());
        }
    }
    else if constexpr (version == 1)
    {
        dim3 threadsPerBlock(32, 1);
        dim3 blocks(M/16, N/16);
        for(int i = 0; i < 4; i++)
        {
            matmul_v1<<<blocks, threadsPerBlock>>>(deviceA[i/2], deviceB[i%2], deviceC[i], M, K, N);
            PRINT_ON_ERROR(cudaGetLastError());
        }
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
        for(int i = 0; i < 4; i++)
        {
            if constexpr (version == 2)
            {
                matmul_v2<BLOCK_SIZE_M, BLOCK_SIZE_N, K_STEP,
                          FRAG_SIZE_M, FRAG_SIZE_K, FRAG_SIZE_N>
                         <<<blocks, threadsPerBlock>>>
                         (deviceA[i/2], deviceB[i%2], deviceC[i], M, K, N);
            }
            else
            {
                matmul_v3<BLOCK_SIZE_M, BLOCK_SIZE_N, K_STEP,
                          FRAG_SIZE_M, FRAG_SIZE_K, FRAG_SIZE_N>
                         <<<blocks, threadsPerBlock>>>
                         (deviceA[i/2], deviceB[i%2], deviceC[i], M, K, N);
            }
            PRINT_ON_ERROR(cudaGetLastError());
        }
    }

    PRINT_ON_ERROR(cudaDeviceSynchronize());

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");

    for(int i = 0; i < 4; i++)
        PRINT_ON_ERROR(cudaMemcpy(hostC[i], deviceC[i], CSize, cudaMemcpyDeviceToHost));

    PROFILE_SEGMENTS_SWITCH("merge");

    for(int i = 0; i < M * N; i++)
        C[i] = hostC[0][i] + hostC[1][i] + hostC[2][i] + hostC[3][i];

    PROFILE_SEGMENTS_SWITCH("free");

    for(int i = 0; i < 2; i++)
    {
        PRINT_ON_ERROR(cudaFree(deviceA[i]));
        PRINT_ON_ERROR(cudaFree(deviceB[i]));
    }
    for(int i = 0; i < 4; i++)
        PRINT_ON_ERROR(cudaFree(deviceC[i]));
    PRINT_ON_ERROR(cudaFree(deviceAFull));
    PRINT_ON_ERROR(cudaFree(deviceBFull));

    PROFILE_SEGMENT_FUNCTION_END();
}

void matmul_simpleMarkidis_v0(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_simpleMarkidis<0>(A, B, C, M, K, N);
}

void matmul_simpleMarkidis_v1(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_simpleMarkidis<1>(A, B, C, M, K, N);
}

void matmul_simpleMarkidis_v2(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_simpleMarkidis<2>(A, B, C, M, K, N);
}

void matmul_simpleMarkidis_v3(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_simpleMarkidis<3>(A, B, C, M, K, N);
}

