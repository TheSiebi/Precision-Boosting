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

template<typename InputType, typename OutputType>
__global__ void matmul_kernel_v0(InputType *A, InputType *B, OutputType *C, int M, int K, int N) 
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    OutputType result = 0.0;
    for (int k = 0; k < K; k++) 
    {
        result += (OutputType)A[m*K + k] * (OutputType)B[k*N + n];
    }
    C[m*N + n] = result;
}

template<typename InputType, typename OutputType,
         int FragSizeM, int FragSizeK, int FragSizeN>
__global__ void matmul_kernel_v1(InputType *A, InputType *B, OutputType *C, int M, int K, int N) 
{
    using namespace nvcuda;

    int warpM = blockIdx.x * FragSizeM;
    int warpN = blockIdx.y * FragSizeN;

    wmma::fragment<wmma::matrix_a, FragSizeM, FragSizeN, FragSizeK, InputType, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, FragSizeM, FragSizeN, FragSizeK, InputType, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, FragSizeM, FragSizeN, FragSizeK, OutputType> cFrag;

    wmma::fill_fragment(cFrag, 0);

    for (int k = 0; k < K; k += FragSizeK) 
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
template<typename InputType, typename OutputType,
         int BlockSizeM, int BlockSizeN, int KStep, 
         int FragSizeM, int FragSizeK, int FragSizeN>
__global__ void matmul_kernel_v2(InputType *A, InputType *B, OutputType *C, int M, int K, int N)
{
    using namespace nvcuda;

    const int scalar_blockMBase = blockIdx.y * BlockSizeM;
    const int scalar_blockNBase = blockIdx.x * BlockSizeN;
    const int scalar_blockBaseA = scalar_blockMBase * K;
    const int scalar_blockBaseB = scalar_blockNBase;
    const int scalar_blockBaseC = scalar_blockMBase * N + scalar_blockNBase;

    const int warpNOffset = threadIdx.y * FragSizeN;
    const int warpMOffset = threadIdx.z * FragSizeM;

    wmma::fragment<wmma::matrix_a, FragSizeM, FragSizeN, FragSizeK, InputType, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, FragSizeM, FragSizeN, FragSizeK, InputType, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, FragSizeM, FragSizeN, FragSizeK, OutputType> cFrag;

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
//blockDim.y == BlockSizeN / (WarpSizeN * FragSizeN)
//blockDim.z == BlockSizeM / (WarpSizeM * FragSizeM)
//gridDim.x == RoundUp(N / BlockSizeN)
//gridDim.y == RoundUp(M / BlockSizeM)
//KStep % WarpSizeK == 0
template<typename InputType, typename OutputType,
         int BlockSizeM, int BlockSizeN, int KStep, 
         int WarpSizeM, int WarpSizeN,
         int FragSizeM, int FragSizeK, int FragSizeN>
__global__ void matmul_kernel_v3(InputType *A, InputType *B, OutputType *C, int M, int K, int N)
{
    using namespace nvcuda;

    const int scalar_blockMBase = blockIdx.y * BlockSizeM;
    const int scalar_blockNBase = blockIdx.x * BlockSizeN;
    const int scalar_blockBaseA = scalar_blockMBase * K;
    const int scalar_blockBaseB = scalar_blockNBase;
    const int scalar_blockBaseC = scalar_blockMBase * N + scalar_blockNBase;

    const int warpNOffset = threadIdx.y * (WarpSizeN * FragSizeN);
    const int warpMOffset = threadIdx.z * (WarpSizeM * FragSizeM);

    wmma::fragment<wmma::matrix_a, FragSizeM, FragSizeN, FragSizeK, InputType, wmma::row_major> aFrag[WarpSizeM];
    wmma::fragment<wmma::matrix_b, FragSizeM, FragSizeN, FragSizeK, InputType, wmma::row_major> bFrag[WarpSizeN];
    wmma::fragment<wmma::accumulator, FragSizeM, FragSizeN, FragSizeK, OutputType> cFrag[WarpSizeM][WarpSizeN];

    for(int m = 0; m < WarpSizeM; m++)
        for(int n = 0; n < WarpSizeN; n++)
            wmma::fill_fragment(cFrag[m][n], 0);

    for (int kBase = 0; kBase < K; kBase += KStep) 
    {
        for(int kOffset = 0; kOffset < KStep; kOffset += FragSizeK)
        {
            int k = kBase + kOffset;
            for(int m = 0; m < WarpSizeM; m++)
            {
                int offsetA = scalar_blockBaseA + (warpMOffset + m * FragSizeM) * K + k;
                wmma::load_matrix_sync(aFrag[m], A + offsetA, K);
            }
            for(int n = 0; n < WarpSizeN; n++)
            {
                int offsetB = scalar_blockBaseB + k * N + warpNOffset + n * FragSizeN;
                wmma::load_matrix_sync(bFrag[n], B + offsetB, N);
            }
            for(int m = 0; m < WarpSizeM; m++)
                for(int n = 0; n < WarpSizeN; n++)
                    wmma::mma_sync(cFrag[m][n], aFrag[m], bFrag[n], cFrag[m][n]);
        }
    }

    int offsetC = scalar_blockBaseC + warpMOffset * N + warpNOffset;
    for(int m = 0; m < WarpSizeM; m++)
        for(int n = 0; n < WarpSizeN; n++)
            wmma::store_matrix_sync(C + offsetC + (m * FragSizeM * N + n * FragSizeN) , cFrag[m][n], N, wmma::mem_row_major);
}


//NOTE(max): Using shared memory like this, causes performance regression. We
//need to investigate why this is the case
//blockDim.x == warpSize
//blockDim.y == BlockSizeN / (WarpSizeN * FragSizeN)
//blockDim.z == BlockSizeM / (WarpSizeM * FragSizeM)
//gridDim.x == RoundUp(N / BlockSizeN)
//gridDim.y == RoundUp(M / BlockSizeM)
//KStep % warpSize == 0
template<typename InputType, typename OutputType,
         int BlockSizeM, int BlockSizeN, int KStep, 
         int WarpSizeM, int WarpSizeN,
         int FragSizeM, int FragSizeK, int FragSizeN>
__global__ void matmul_kernel_v4(InputType *A, InputType *B, OutputType *C, int M, int K, int N)
{
    using namespace nvcuda;

    __shared__ InputType SharedA[BlockSizeM][KStep];
    __shared__ InputType SharedB[KStep][BlockSizeN];

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

    wmma::fragment<wmma::matrix_a, FragSizeM, FragSizeN, FragSizeK, InputType, wmma::row_major> aFrag[WarpSizeM];
    wmma::fragment<wmma::matrix_b, FragSizeM, FragSizeN, FragSizeK, InputType, wmma::row_major> bFrag[WarpSizeN];
    wmma::fragment<wmma::accumulator, FragSizeM, FragSizeN, FragSizeK, OutputType> cFrag[WarpSizeM][WarpSizeN];

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
            SharedA[m][loadAKOffset] = A[loadABase + m * K + loadAKOffset];
        }
        for(int kOffset = 0; kOffset < KStep; kOffset += loadBRowStep)
        {
            int k = kOffset + loadBKOffset;
            SharedB[k][loadBNOffset] = B[loadBBase + k * N + loadBNOffset];
        }

        __syncthreads();

        for(int kOffset = 0; kOffset < KStep; kOffset += FragSizeK)
        {
            for(int m = 0; m < WarpSizeM; m++)
            {
                InputType *aOffset = &SharedA[warpMOffset + m * FragSizeM][kOffset];
                wmma::load_matrix_sync(aFrag[m], aOffset, KStep);
            }
            for(int n = 0; n < WarpSizeN; n++)
            {
                InputType *bOffset = &SharedB[kOffset][warpNOffset + n * FragSizeN];
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


template <const int BM, const int BN, const int BK, const int WM, const int WN, 
          const int N_WARP_ROWS_PER_BLOCK, const int N_WARP_COLS_PER_BLOCK,
          const int WNITER, const int WMITER, const int TM, const int TN>
__global__ void matmul_kernel_float_v0(const float *A, const float *B, float *C, int M, int K, int N)
{
    // allocate space for the current blocktile in shared memory
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beggining of A's row and B's column
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // warpID in threadBlock
    const int warpID = threadIdx.x / WARP_SIZE;
    // The indices this warp has in the block tile
    const int warpRow = warpID / N_WARP_COLS_PER_BLOCK;
    const int warpCol = warpID % N_WARP_COLS_PER_BLOCK;

    // size of warp subtile
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    // Placement of the thread in the warp subtile
    


    // Loads are vectorized and each thread will load 4 elements into SMEM
    const int elemsPerThread = 4;
    // Calculate indices that this thread will load from GMEM to SMEM
    // Note that for coalescing, it's important that consecutive threadIDs
    // access consecutive memory addresses
    const int innerRowA = threadIdx.x / (BK / elemsPerThread);
    const int innerColA = threadIdx.x % (BK / elemsPerThread);
    const int innerRowB = threadIdx.x / (BN / elemsPerThread);
    const int innerColB = threadIdx.x % (BN / elemsPerThread);
    // complete #rows that gets loaded in one loading iteration
    const int rowStrideA = elemsPerThread * blockDim.x / BK;
    const int rowStrideB = elemsPerThread * blockDim.x / BN;

    const int loadIterationsA = BM * BK / (elemsPerThread * blockDim.x);
    const int loadIterationsB = BK * BN / (elemsPerThread * blockDim.x);

    // Loop over all block tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // populate SMEM cache
        for (int i = 0; i < loadIterationsA; i++)
        {
            loadAndSplit<float4, float>(A, K, innerRowA, i * rowStrideA, innerColA, As, dAs, BK);
        }
        for (int i = 0; i < loadIterationsB; i++)
        {
            loadAndSplit<float4, float>(B, N, innerRowB, i * rowStrideB, innerColB, Bs, dBs, BN);
        }

        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // start of data belonging to respective warp
        int warpOffsetA = warpRow * WM * BK;
        int warpOffsetB = warpCol * WN;
        
        // Warning: if this is removed and the compiler unrolls this loop, register usage for this kernel is too high
        // and it can't be launched
        #pragma unroll 1
        for (int chunk = 0; chunk < CHUNK_K; chunk++)
        {   
            for (int tileCol = 0; tileCol < N_WMMA_COLS_PER_WARP; tileCol++)
            {
                wmma::load_matrix_sync(bFrag[tileCol], Bs + warpOffsetB + tileCol * WMMA_N + chunk * WMMA_K * BN, BN);
                wmma::load_matrix_sync(dbFrag[tileCol], dBs + warpOffsetB + tileCol * WMMA_N + chunk * WMMA_K * BN, BN);
            }
            // calculate mmul
            for (int tileRow = 0; tileRow < N_WMMA_ROWS_PER_WARP; tileRow++)
            {
                wmma::load_matrix_sync(aFrag, As + warpOffsetA + tileRow * WMMA_M * BK + chunk * WMMA_K, BK);
                wmma::load_matrix_sync(daFrag, dAs + warpOffsetA + tileRow * WMMA_M * BK + chunk * WMMA_K, BK);
                for (int tileCol = 0; tileCol < N_WMMA_COLS_PER_WARP; tileCol++)
                {
                    wmma::fill_fragment(tmpFrag, 0.0f);
                    wmma::mma_sync(tmpFrag, aFrag, bFrag[tileCol], tmpFrag);

                    // accumulate outside tensor cores to avoid RZ
                    for (int i = 0; i < cFrag[tileRow][tileCol].num_elements; i++)
                    {
                        cFrag[tileRow][tileCol].x[i] += tmpFrag.x[i];
                    }
                    
                    // matrices for error correction can be directly accumulated in the tensor core
                    wmma::mma_sync(dcFrag[tileRow][tileCol], daFrag, bFrag[tileCol], dcFrag[tileRow][tileCol]);
                    wmma::mma_sync(dcFrag[tileRow][tileCol], aFrag, dbFrag[tileCol], dcFrag[tileRow][tileCol]);
                }
            }
        }

        __syncthreads();
    }

    // Store results back to C matrix
    float *warpC = &C[warpRow * WM * N + warpCol * WN];
    for (int tileRow = 0; tileRow < N_WMMA_ROWS_PER_WARP; tileRow++)
    {
        for (int tileCol = 0; tileCol < N_WMMA_COLS_PER_WARP; tileCol++)
        {   
            // perform merge according to Ootomo paper
            for (int i = 0; i < cFrag[tileRow][tileCol].num_elements; i++)
            {
                cFrag[tileRow][tileCol].x[i] += dcFrag[tileRow][tileCol].x[i] / 2048.0f;
            }

            wmma::store_matrix_sync(warpC + tileCol * WMMA_N, cFrag[tileRow][tileCol], N, wmma::mem_row_major);
        }
        warpC += WMMA_M * N;
    }
}


template<typename InputType, typename OutputType, int version>
void matmul(InputType *A, InputType *B, OutputType *C, int M, int K, int N)
{
    const int BLOCK_SIZE_M = std::is_same<InputType, half>::value ? 64 : 32;
    const int BLOCK_SIZE_N = std::is_same<InputType, half>::value ? 64 : 32;
    const int K_STEP       = 16;
    const int WARP_SIZE_M  = 2;
    const int WARP_SIZE_N  = 2;
    const int FRAG_SIZE_M  = std::is_same<InputType, half>::value ? 16 : 8;
    const int FRAG_SIZE_K  = std::is_same<InputType, half>::value ? 16 : 4;
    const int FRAG_SIZE_N  = std::is_same<InputType, half>::value ? 16 : 8;
    if constexpr (version == 0)
    {
        dim3 threadsPerBlock(16, 16);
        dim3 blocks(M/threadsPerBlock.x, N/threadsPerBlock.y);
        matmul_kernel_v0<InputType, OutputType>
                        <<<blocks, threadsPerBlock>>>(A, B, C, M, K, N);
    }
    else if constexpr (version == 1)
    {
        dim3 threadsPerBlock(32, 1);
        dim3 blocks(M/FRAG_SIZE_M, N/FRAG_SIZE_N);
        matmul_kernel_v1<InputType, OutputType,
                         FRAG_SIZE_M, FRAG_SIZE_K, FRAG_SIZE_N>
                        <<<blocks, threadsPerBlock>>>(A, B, C, M, K, N);
    }
    else if constexpr (version == 2)
    {
        dim3 threadsPerBlock(WARP_SIZE, BLOCK_SIZE_N / FRAG_SIZE_N, BLOCK_SIZE_M / FRAG_SIZE_M);
        dim3 blocks(DivRoundUp(N, BLOCK_SIZE_N), DivRoundUp(M, BLOCK_SIZE_M));
        matmul_kernel_v2<InputType, OutputType,
                  BLOCK_SIZE_M, BLOCK_SIZE_N, K_STEP,
                  FRAG_SIZE_M, FRAG_SIZE_K, FRAG_SIZE_N>
                 <<<blocks, threadsPerBlock>>>
                 (A, B, C, M, K, N);
    }
    else if constexpr (version == 3 || version == 4)
    {
        dim3 threadsPerBlock(WARP_SIZE, 
                             BLOCK_SIZE_N / (WARP_SIZE_N * FRAG_SIZE_N), 
                             BLOCK_SIZE_M / (WARP_SIZE_M * FRAG_SIZE_M));
        dim3 blocks(DivRoundUp(N, BLOCK_SIZE_N), DivRoundUp(M, BLOCK_SIZE_M));
        if constexpr (version == 3)
        {
            matmul_kernel_v3<InputType, OutputType,
                      BLOCK_SIZE_M, BLOCK_SIZE_N, K_STEP,
                      WARP_SIZE_M, WARP_SIZE_N,
                      FRAG_SIZE_M, FRAG_SIZE_K, FRAG_SIZE_N>
                     <<<blocks, threadsPerBlock>>>
                     (A, B, C, M, K, N);
        }
        else
        {
            matmul_kernel_v4<InputType, OutputType,
                      BLOCK_SIZE_M, BLOCK_SIZE_N, K_STEP,
                      WARP_SIZE_M, WARP_SIZE_N,
                      FRAG_SIZE_M, FRAG_SIZE_K, FRAG_SIZE_N>
                     <<<blocks, threadsPerBlock>>>
                     (A, B, C, M, K, N);
        }
    }
    PRINT_ON_ERROR(cudaGetLastError());
}

template void matmul<half, float, 0>(half*, half*, float*, int, int, int);
template void matmul<half, float, 1>(half*, half*, float*, int, int, int);
template void matmul<half, float, 2>(half*, half*, float*, int, int, int);
template void matmul<half, float, 3>(half*, half*, float*, int, int, int);
template void matmul<half, float, 4>(half*, half*, float*, int, int, int);

template<typename InputType, typename OutputType, int version>
flop_counts matmul_cuda(InputType *A, InputType *B, OutputType *C, int M, int K, int N) 
{
    assert((M & 0xF) == 0);
    assert((K & 0xF) == 0);
    assert((N & 0xF) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate");
    size_t ASize = M * K * sizeof(InputType);
    size_t BSize = K * N * sizeof(InputType);
    size_t CSize = M * N * sizeof(OutputType);

    InputType *deviceA, *deviceB;
    OutputType *deviceC;
    PRINT_ON_ERROR(cudaMalloc(&deviceA, ASize));
    PRINT_ON_ERROR(cudaMalloc(&deviceB, BSize));
    PRINT_ON_ERROR(cudaMalloc(&deviceC, CSize));

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    PRINT_ON_ERROR(cudaMemcpy(deviceA, A, ASize, cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(deviceB, B, BSize, cudaMemcpyHostToDevice));

    PROFILE_SEGMENTS_SWITCH("matmul");
    matmul<InputType, OutputType, version>(deviceA, deviceB, deviceC, M, K, N);
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
#if SM_VERSION >= 800
template flop_counts matmul_cuda<double, double, 0>(double*, double*, double*, int, int, int);
template flop_counts matmul_cuda<double, double, 1>(double*, double*, double*, int, int, int);
template flop_counts matmul_cuda<double, double, 2>(double*, double*, double*, int, int, int);
template flop_counts matmul_cuda<double, double, 3>(double*, double*, double*, int, int, int);
template flop_counts matmul_cuda<double, double, 4>(double*, double*, double*, int, int, int);
#endif

