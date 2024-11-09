#include <assert.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../matmul.h"
#include "../profiler.h"

/**
 * Note: Kernels in this file have been inspired by: 
 *  - https://github.com/siboehm/SGEMM_CUDA/tree/master
 *  - https://github.com/Bruce-Lee-LY/cuda_hgemm/blob/master/src/wmma/wmma_base.cu#L86
 */


#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32

struct matmulTemplateArgs
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

struct matmulScales
{
    int scaleBM;
    int scaleBN;
    int scaleChunk;
    int scaleWM;
    int scaleWN;
};

constexpr struct matmulScales getArgScales(int minSize) {
    
    if (minSize == 16)
    {
        return {1, 1, 1, 1, 1};
    }
    else if (minSize == 256)
    {
        return {8, 8, 2, 2, 2};
    }
    // Warning: Before adding new configuration, make sure
    // that the shared memory of the GPU is large enough to handle the block dimensions

    return {-1, -1, -1, -1, -1};
}

template<int minSize>
constexpr struct matmulTemplateArgs getMatmulTemplateArgs()
{   
    constexpr struct matmulScales scales = getArgScales(minSize);
    constexpr int CHUNK_K = scales.scaleChunk;
    constexpr int BM = WMMA_M * scales.scaleBM;
    constexpr int BN = WMMA_N * scales.scaleBN;
    constexpr int BK = WMMA_K * CHUNK_K;
    
    constexpr int WM = WMMA_M * scales.scaleWM;
    constexpr int WN = WMMA_N * scales.scaleWN;

    // the "warpdimensions" must divide the block dimensions
    static_assert(BM % WM == 0);
    static_assert(BN % WN == 0);
    constexpr int N_WARP_ROWS_PER_BLOCK = BM / WM;
    constexpr int N_WARP_COLS_PER_BLOCK = BN / WN;
    constexpr int N_WMMA_ROWS_PER_WARP = WM / WMMA_M;
    constexpr int N_WMMA_COLS_PER_WARP = WN / WMMA_N;
    constexpr int threadsPerBlock = N_WARP_ROWS_PER_BLOCK * N_WARP_COLS_PER_BLOCK * WARP_SIZE;
    // In each SMEM loading iteration, each thread loads 4 values from GMEM
    // These asserts ensures that the loading loop does not convert divergent branches (i.e. each thread has 
    // the same amount of values to load)
    static_assert((BM * BK) % (4 * threadsPerBlock) == 0);
    static_assert((BK * BN) % (4 * threadsPerBlock) == 0);
    // These asserts ensure that in each SMEM loading iteration, the threads load N entire rows (and not a half row or something)
    // of the shared memory
    static_assert((4 * threadsPerBlock) % BK == 0);
    static_assert((4 * threadsPerBlock) % BN == 0);

    return {BM, BN, BK, WM, WN, CHUNK_K, N_WARP_ROWS_PER_BLOCK, N_WARP_COLS_PER_BLOCK, N_WMMA_ROWS_PER_WARP, N_WMMA_COLS_PER_WARP, threadsPerBlock};
}

/**
 * Given floats x, y, z, w, we have:
 *  - x = x.x + dx.x / 2048
 *  - y = x.y + dx.y / 2048
 *  - z = y.x + dy.x / 2048
 *  - w = y.y + dy.y / 2048
 */
struct split
{
    // original terms
    half2 x;
    half2 y;
    // error terms
    half2 dx;
    half2 dy;
};

/**
 * Perform vectorized split of a float4 into 8 halfs according to the Ootomo paper
 */
__device__ struct split split_Ootomo(float4 value)
{
    float2 first = make_float2(value.x, value.y);
    float2 second = make_float2(value.z, value.w);
    struct split split;

    split.x = __float22half2_rn(first);
    split.y = __float22half2_rn(second);

    float2 reconstructed = __half22float2(split.x);
    split.dx = __float22half2_rn(make_float2((first.x - reconstructed.x) * 2048, (first.y - reconstructed.y) * 2048));

    reconstructed = __half22float2(split.y);
    split.dy = __float22half2_rn(make_float2((second.x - reconstructed.x) * 2048, (second.y - reconstructed.y) * 2048));

    return split;
}   

/**
 * Simple kernel that splits a float matrix into two half matrices according to the Ootoma paper
 */
__global__ void split_cuda(float *A, half *A0, half *A1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float value = A[i];
    half mainPart = (half)value;
    A0[i] = mainPart;
    A1[i] = (half)((value - (float)mainPart) * 2048.0f);
}

/**
 * Simple kernel that performs the merge described in the Ootomo paper including the smallest term. 
 */
__global__ void merge_cuda(float *C, float *AB, float *dAB, float *AdB, float *dAdB)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = AB[i] + (dAB[i] + AdB[i]) / 2048.0f + dAdB[i] / 4194304.0f;
}

/**
 * Kernel that performs half precision matrix multiplication using tensore cores. Does not implement
 * specific Ootomo logic. In particular, it does NOT do the accumulation of values outside the tensor 
 * cores to avoid RZ.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN, const int CHUNK_K,
          const int N_WARP_ROWS_PER_BLOCK,
          const int N_WARP_COLS_PER_BLOCK,
          const int N_WMMA_ROWS_PER_WARP,
          const int N_WMMA_COLS_PER_WARP>
__global__ void matmul_v0_kernel(const half *A, const half *B, float *C, int M, int K, int N)
{
    using namespace nvcuda;

    // allocate space for the current blocktile in shared memory
    __shared__ half As[BM * BK];
    __shared__ half Bs[BK * BN];

    // Move blocktile to beggining of A's row and B's column
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // warpID in threadBlock
    const int warpID = threadIdx.x / WARP_SIZE;

    // The indices this warp has in the block tile
    const int warpRow = warpID / N_WARP_COLS_PER_BLOCK;
    const int warpCol = warpID % N_WARP_COLS_PER_BLOCK;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag[N_WMMA_ROWS_PER_WARP][N_WMMA_COLS_PER_WARP];

    for (int i = 0; i < N_WMMA_ROWS_PER_WARP; i++)
    {
        for (int j = 0; j < N_WMMA_COLS_PER_WARP; j++)
        {
            wmma::fill_fragment(cFrag[i][j], 0.0f);
        }
    }

    // Calculate indices that this thread will load from GMEM to SMEM
    // Loads are vectorized and each thread will load 4 elements into SMEM
    // Note that for coalescing, it's important that consecutive threadIDs
    // access consecutive memory addresses
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    // complete #rows that gets loaded in one loading iteration
    const int strideA = 4 * blockDim.x / BK;
    const int strideB = 4 * blockDim.x / BN;

    // Loop over all block tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // populate SMEM cache using vectorized loads
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA)
        {
            reinterpret_cast<half2 *>(&As[(innerRowA + loadOffset) * BK + innerColA * 4])[0] = 
                reinterpret_cast<const half2 *>(&A[(innerRowA + loadOffset) * K + innerColA * 4])[0];
            reinterpret_cast<half2 *>(&As[(innerRowA + loadOffset) * BK + innerColA * 4 + 2])[0] = 
                reinterpret_cast<const half2 *>(&A[(innerRowA + loadOffset) * K + innerColA * 4 + 2])[0];
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB)
        {
            reinterpret_cast<half2 *>(&Bs[(innerRowB + loadOffset) * BN + innerColB * 4])[0] = 
                reinterpret_cast<const half2 *>(&B[(innerRowB + loadOffset) * N + innerColB * 4])[0];
            reinterpret_cast<half2 *>(&Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + 2])[0] = 
                reinterpret_cast<const half2 *>(&B[(innerRowB + loadOffset) * N + innerColB * 4 + 2])[0];
        }

        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // start of data belonging to respective warp
        half *warpAs = &As[warpRow * WM * BK];
        half *warpBs = &Bs[warpCol * WN];
        
        // calculate mmul
        for (int tileRow = 0; tileRow < N_WMMA_ROWS_PER_WARP; tileRow++)
        {
            for (int tileCol = 0; tileCol < N_WMMA_COLS_PER_WARP; tileCol++)
            {
                for (int chunk = 0; chunk < CHUNK_K; chunk++)
                {
                    wmma::load_matrix_sync(aFrag, warpAs + chunk * WMMA_K, BK);
                    wmma::load_matrix_sync(bFrag, warpBs + chunk * WMMA_K * BN, BN);

                    wmma::mma_sync(cFrag[tileRow][tileCol], aFrag, bFrag, cFrag[tileRow][tileCol]);
                }
                warpBs += WMMA_N;
            }
            warpBs = &Bs[warpCol * WN];
            warpAs += WMMA_M * BK;
        }
    }

    // Store results back to C matrix
    float *warpC = &C[warpRow * WM * N + warpCol * WN];
    for (int tileRow = 0; tileRow < N_WMMA_ROWS_PER_WARP; tileRow++)
    {
        for (int tileCol = 0; tileCol < N_WMMA_COLS_PER_WARP; tileCol++)
        {
            wmma::store_matrix_sync(warpC + tileCol * WMMA_N, cFrag[tileRow][tileCol], N, wmma::mem_row_major);
        }
        warpC += WMMA_M * N;
    }
}

// Code adapted from: https://github.com/siboehm/SGEMM_CUDA/tree/master
template <const int BM, const int BN, const int BK, const int WM, const int WN, const int CHUNK_K,
          const int N_WARP_ROWS_PER_BLOCK,
          const int N_WARP_COLS_PER_BLOCK,
          const int N_WMMA_ROWS_PER_WARP,
          const int N_WMMA_COLS_PER_WARP>
__global__ void matmul_v1_kernel(const float *A, const float *B, float *C, int M, int K, int N)
{
    using namespace nvcuda;

    // allocate space for the current blocktile in shared memory
    __shared__ half As[BM * BK];
    __shared__ half Bs[BK * BN];
    __shared__ half dAs[BM * BK];
    __shared__ half dBs[BK * BN];

    // Move blocktile to beggining of A's row and B's column
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // warpID in threadBlock
    const int warpID = threadIdx.x / WARP_SIZE;
    // thread LaneID in warp
    // const int laneID = threadIdx.x % WARP_SIZE;
    // The indices this warp has in the block tile
    const int warpRow = warpID / N_WARP_COLS_PER_BLOCK;
    const int warpCol = warpID % N_WARP_COLS_PER_BLOCK;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag[N_WMMA_ROWS_PER_WARP][N_WMMA_COLS_PER_WARP];

    for (int i = 0; i < N_WMMA_ROWS_PER_WARP; i++)
    {
        for (int j = 0; j < N_WMMA_COLS_PER_WARP; j++)
        {
            wmma::fill_fragment(cFrag[i][j], 0.0f);
        }
    }

    // Calculate indices that this thread will load from GMEM to SMEM
    // Loads are vectorized and each thread will load 4 elements into SMEM
    // Note that for coalescing, it's important that consecutive threadIDs
    // access consecutive memory addresses
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    // complete #rows that gets loaded in one loading iteration
    const int strideA = 4 * blockDim.x / BK;
    const int strideB = 4 * blockDim.x / BN;

    // Loop over all block tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // populate SMEM cache using vectorized loads
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA)
        {
            float4 tmp = *(reinterpret_cast<const float4 *>(A + (innerRowA + loadOffset) * K + innerColA * 4));
            struct split tmp_split = split_Ootomo(tmp);
            As[(innerRowA + loadOffset) * BK + innerColA * 4 + 0] = tmp_split.x.x;
            As[(innerRowA + loadOffset) * BK + innerColA * 4 + 1] = tmp_split.x.y;
            As[(innerRowA + loadOffset) * BK + innerColA * 4 + 2] = tmp_split.y.x;
            As[(innerRowA + loadOffset) * BK + innerColA * 4 + 3] = tmp_split.y.y;
            dAs[(innerRowA + loadOffset) * BK + innerColA * 4 + 0] = tmp_split.dx.x;
            dAs[(innerRowA + loadOffset) * BK + innerColA * 4 + 1] = tmp_split.dx.y;
            dAs[(innerRowA + loadOffset) * BK + innerColA * 4 + 2] = tmp_split.dy.x;
            dAs[(innerRowA + loadOffset) * BK + innerColA * 4 + 3] = tmp_split.dy.y;
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB)
        {
            float4 tmp = *(reinterpret_cast<const float4 *>(B + (innerRowB + loadOffset) * N + innerColB * 4));
            struct split tmp_split = split_Ootomo(tmp);
            Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + 0] = tmp_split.x.x;
            Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + 1] = tmp_split.x.y;
            Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + 2] = tmp_split.y.x;
            Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + 3] = tmp_split.y.y;
            dBs[(innerRowB + loadOffset) * BN + innerColB * 4 + 0] = tmp_split.dx.x;
            dBs[(innerRowB + loadOffset) * BN + innerColB * 4 + 1] = tmp_split.dx.y;
            dBs[(innerRowB + loadOffset) * BN + innerColB * 4 + 2] = tmp_split.dy.x;
            dBs[(innerRowB + loadOffset) * BN + innerColB * 4 + 3] = tmp_split.dy.y;
        }

        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // start of data belonging to respective warp
        half *warpAs = &As[warpRow * WM * BK];
        half *warpBs = &Bs[warpCol * WN];

        for (int tileRow = 0; tileRow < N_WMMA_ROWS_PER_WARP; tileRow++)
        {
            for (int tileCol = 0; tileCol < N_WMMA_COLS_PER_WARP; tileCol++)
            {
                for (int chunk = 0; chunk < CHUNK_K; chunk++)
                {
                    wmma::load_matrix_sync(aFrag, warpAs + CHUNK_K * WMMA_K, BK);
                    wmma::load_matrix_sync(bFrag, warpBs + CHUNK_K * WMMA_K * BN, BN);

                    wmma::mma_sync(cFrag[tileRow][tileCol], aFrag, bFrag, cFrag[tileRow][tileCol]);
                }
                warpBs += WMMA_N;
            }
            warpAs += WMMA_M * BK;
        }
    }
}

template<int version>
void matmul_Oootomo(float *A, float *B, float *C, int M, int K, int N) 
{
    assert((M % 16) == 0);
    assert((K % 16) == 0);
    assert((N % 16) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate gpu");
    
    int AElems = M * K;
    int BElems = K * N;
    int CElems = M * N;
    float *deviceAFull, *deviceBFull, *deviceCFull;
    cudaMalloc(&deviceAFull, AElems * sizeof(float));
    cudaMalloc(&deviceBFull, BElems * sizeof(float));
    cudaMalloc(&deviceCFull, CElems * sizeof(float));

    // these identifiers are just outside the if because otherwise, compilation does not work
    half *deviceA[2], *deviceB[2];
    // {AB, dAB, AdB, dAdB}
    float *deviceC[4];
    if constexpr(version == 0)
    {
        for(int i = 0; i < 2; i++)
        {
            cudaMalloc(&deviceA[i], AElems * sizeof(half));
            cudaMalloc(&deviceB[i], BElems * sizeof(half));
        }
        for(int i = 0; i < 4; i++)
            cudaMalloc(&deviceC[i], CElems * sizeof(float));
    }

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    cudaMemcpy(deviceAFull, A, AElems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBFull, B, BElems * sizeof(float), cudaMemcpyHostToDevice);

    if constexpr(version == 0)
    {
        PROFILE_SEGMENTS_SWITCH("split");
        int threadsPerBlock = 256;
        split_cuda<<<M * K / threadsPerBlock, threadsPerBlock>>>(deviceAFull, deviceA[0], deviceA[1]);
        split_cuda<<<K * N / threadsPerBlock, threadsPerBlock>>>(deviceBFull, deviceB[0], deviceB[1]);

        cudaDeviceSynchronize();
    }

    PROFILE_SEGMENTS_SWITCH("matmul");
    if constexpr(version == 0)
    {
        if (M < 256 | K < 256 | N < 256)
        {
            constexpr struct matmulTemplateArgs p = getMatmulTemplateArgs<16>();
            assert(M % p.BM == 0);
            assert(N % p.BN == 0);
            assert(K % p.BK == 0);

            dim3 blocks(M / p.BM, N / p.BN);
            matmul_v0_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                <<<blocks, p.threadsPerBlock>>>(deviceA[0], deviceB[0], deviceC[0], M, K, N);
            matmul_v0_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                <<<blocks, p.threadsPerBlock>>>(deviceA[1], deviceB[0], deviceC[1], M, K, N);
            matmul_v0_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                <<<blocks, p.threadsPerBlock>>>(deviceA[0], deviceB[1], deviceC[2], M, K, N);
            matmul_v0_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                <<<blocks, p.threadsPerBlock>>>(deviceA[1], deviceB[1], deviceC[3], M, K, N);
        } 
        else 
        {
            constexpr struct matmulTemplateArgs p = getMatmulTemplateArgs<256>();
            assert(M % p.BM == 0);
            assert(N % p.BN == 0);
            assert(K % p.BK == 0);

            dim3 blocks(M / p.BM, N / p.BN);
            matmul_v0_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                <<<blocks, p.threadsPerBlock>>>(deviceA[0], deviceB[0], deviceC[0], M, K, N);
            matmul_v0_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                <<<blocks, p.threadsPerBlock>>>(deviceA[1], deviceB[0], deviceC[1], M, K, N);
            matmul_v0_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                <<<blocks, p.threadsPerBlock>>>(deviceA[0], deviceB[1], deviceC[2], M, K, N);
            matmul_v0_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                <<<blocks, p.threadsPerBlock>>>(deviceA[1], deviceB[1], deviceC[3], M, K, N);
        }
    } 
    else if constexpr(version == 1)
    {
        // ...
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    if constexpr(version == 0)
    {
        PROFILE_SEGMENTS_SWITCH("merge");
        int threadsPerBlock = 256;
        merge_cuda<<<M * K / threadsPerBlock, threadsPerBlock>>>(deviceCFull, deviceC[0], deviceC[1], deviceC[2], deviceC[3]);

        cudaDeviceSynchronize();
    }

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    cudaMemcpy(C, deviceCFull, CElems * sizeof(float), cudaMemcpyDeviceToHost);


    PROFILE_SEGMENTS_SWITCH("free");
    cudaFree(deviceAFull);
    cudaFree(deviceBFull);
    cudaFree(deviceCFull);

    if constexpr(version == 0)
    {
        for(int i = 0; i < 2; i++)
        {
            cudaFree(deviceA[i]);
            cudaFree(deviceB[i]);
        }
        for(int i = 0; i < 4; i++)
            cudaFree(deviceC[i]);
    }

    PROFILE_SEGMENT_FUNCTION_END();
}

void matmul_Oootomo_v0(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_Oootomo<0>(A, B, C, M, K, N);
}

void matmul_Oootomo_v1(float *A, float *B, float *C, int M, int K, int N)
{
    matmul_Oootomo<1>(A, B, C, M, K, N);
}
