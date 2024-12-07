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
#include "../timer.h"
#include "./split_merge_cuda.h"

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

constexpr struct matmulScales getArgScales(int configuration) {
    
    if (configuration == 0)
    {
        return {1, 1, 1, 1, 1};
    }
    else if (configuration == 1)
    {
        return {8, 8, 2, 2, 2};
    } 
    else if (configuration == 2)
    {
        return {4, 4, 2, 2, 2};
    }
    // Warning: Before adding new configuration, make sure
    // that the shared memory of the GPU is large enough to handle the block dimensions

    return {-1, -1, -1, -1, -1};
}

template<int configuration>
constexpr struct matmulTemplateArgs getMatmulTemplateArgs()
{   
    constexpr struct matmulScales scales = getArgScales(configuration);
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
    // These asserts ensures that the loading loop does not create divergent branches (i.e. each thread has 
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
 * Kernel that performs half precision matrix multiplication using tensore cores. Does not implement
 * specific Ootomo logic. In particular, it does NOT do the accumulation of values outside the tensor 
 * cores to avoid RZ.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN, const int CHUNK_K,
          const int N_WARP_ROWS_PER_BLOCK,
          const int N_WARP_COLS_PER_BLOCK,
          const int N_WMMA_ROWS_PER_WARP,
          const int N_WMMA_COLS_PER_WARP>
__global__ void matmul_v0_kernel(const half *A, const half *B, float *C, size_t M, size_t K, size_t N)
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

    
    // Loads are vectorized and each thread will load elemsPerThread elements into SMEM   
    const int elemsPerThread = 2;
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
        // populate SMEM cache using vectorized loads
        for (int i = 0; i < loadIterationsA; i++)
        {
            reinterpret_cast<half2 *>(&As[(innerRowA + i * rowStrideA) * BK + innerColA * elemsPerThread])[0] = 
                reinterpret_cast<const half2 *>(&A[(innerRowA + i * rowStrideA) * K + innerColA * elemsPerThread])[0];
        }
        for (int i = 0; i < loadIterationsB; i++)
        {
            reinterpret_cast<half2 *>(&Bs[(innerRowB + i * rowStrideB) * BN + innerColB * elemsPerThread])[0] = 
                reinterpret_cast<const half2 *>(&B[(innerRowB + i * rowStrideB) * N + innerColB * elemsPerThread])[0];
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

        __syncthreads();
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
__device__ __forceinline__ struct split split_Ootomo(float4 value)
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
 * Performs a vectorized 4 element load from GMEM, splits the obtained values into 8 halfs according to the Ootomo paper
 * and saves them to SMEM
 */
template<typename loadType, typename matType>
__device__ __forceinline__ void loadAndSplit(const matType *A, int ACols, int innerRow, int rowOffset, int innerCol, half *As, half *dAs, int AsCols)
{
    loadType tmp = *(reinterpret_cast<const loadType *>(A + (innerRow + rowOffset) * ACols + innerCol * 4));
    struct split tmp_split = split_Ootomo(tmp);
    reinterpret_cast<float2 *>(&As[(innerRow + rowOffset) * AsCols + innerCol * 4])[0] = 
        reinterpret_cast<float2 *>(&tmp_split)[0];
    reinterpret_cast<float2 *>(&dAs[(innerRow + rowOffset) * AsCols + innerCol * 4])[0] = 
        reinterpret_cast<float2 *>(&tmp_split.dx)[0];
    // As[(innerRow + rowOffset) * AsCols + innerCol * 4 + 0] = tmp_split.x.x;
    // As[(innerRow + rowOffset) * AsCols + innerCol * 4 + 1] = tmp_split.x.y;
    // As[(innerRow + rowOffset) * AsCols + innerCol * 4 + 2] = tmp_split.y.x;
    // As[(innerRow + rowOffset) * AsCols + innerCol * 4 + 3] = tmp_split.y.y;
    // dAs[(innerRow + rowOffset) * AsCols + innerCol * 4 + 0] = tmp_split.dx.x;
    // dAs[(innerRow + rowOffset) * AsCols + innerCol * 4 + 1] = tmp_split.dx.y;
    // dAs[(innerRow + rowOffset) * AsCols + innerCol * 4 + 2] = tmp_split.dy.x;
    // dAs[(innerRow + rowOffset) * AsCols + innerCol * 4 + 3] = tmp_split.dy.y;

    // assert(fabs(((float)tmp_split.x.x + (float)tmp_split.dx.x / 2048.0f) - tmp.x) < 0.001f);
    // assert(fabs(((float)tmp_split.x.y + (float)tmp_split.dx.y / 2048.0f) - tmp.y) < 0.001f);
    // assert(fabs(((float)tmp_split.y.x + (float)tmp_split.dy.x / 2048.0f) - tmp.z) < 0.001f);
    // assert(fabs(((float)tmp_split.y.y + (float)tmp_split.dy.y / 2048.0f) - tmp.w) < 0.001f);  
}

/**
 * Performs a vectorized 4 element load from GMEM, splits the obtained values into 8 halfs according to the Ootomo paper
 * and saves them to SMEM
 */
template<typename loadType, typename matType>
__device__ __forceinline__ void loadColMajorAndSplit(const matType *A, int ACols, int innerRow, int rowOffset, int innerCol, half *As, half *dAs, int AsRows)
{
    loadType tmp = *(reinterpret_cast<const loadType *>(A + (innerRow + rowOffset) * ACols + innerCol * 4));
    struct split tmp_split = split_Ootomo(tmp);
    As[(innerCol * 4 + 0) * AsRows + innerRow + rowOffset] = tmp_split.x.x;
    As[(innerCol * 4 + 1) * AsRows + innerRow + rowOffset] = tmp_split.x.y;
    As[(innerCol * 4 + 2) * AsRows + innerRow + rowOffset] = tmp_split.y.x;
    As[(innerCol * 4 + 3) * AsRows + innerRow + rowOffset] = tmp_split.y.y;
    dAs[(innerCol * 4 + 0) * AsRows + innerRow + rowOffset] = tmp_split.dx.x;
    dAs[(innerCol * 4 + 1) * AsRows + innerRow + rowOffset] = tmp_split.dx.y;
    dAs[(innerCol * 4 + 2) * AsRows + innerRow + rowOffset] = tmp_split.dy.x;
    dAs[(innerCol * 4 + 3) * AsRows + innerRow + rowOffset] = tmp_split.dy.y;
}

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int CHUNK_K,
          const int N_WARP_ROWS_PER_BLOCK,
          const int N_WARP_COLS_PER_BLOCK,
          const int N_WMMA_ROWS_PER_WARP,
          const int N_WMMA_COLS_PER_WARP>
__global__ void matmul_v1_kernel(const float *A, const float *B, float *C, size_t M, size_t K, size_t N)
{
    using namespace nvcuda;

    // allocate space for the current blocktile in shared memory
    // shared_mem_fp16 smem_a, smem_b, smem_da, smem_db
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

    // fragment frag_a, frag_da, frag_b, frag_db, frag_c, frad_dc, frag_tmp 
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> daFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> dbFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag[N_WMMA_ROWS_PER_WARP][N_WMMA_COLS_PER_WARP];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dcFrag[N_WMMA_ROWS_PER_WARP][N_WMMA_COLS_PER_WARP];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> tmpFrag;

    for (int i = 0; i < N_WMMA_ROWS_PER_WARP; i++)
    {
        for (int j = 0; j < N_WMMA_COLS_PER_WARP; j++)
        {
            wmma::fill_fragment(cFrag[i][j], 0.0f);
            wmma::fill_fragment(dcFrag[i][j], 0.0f);
        }
    }

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
            // smem_a = toFP16(mem_a[k]), smem_da = toFP16((mem_a[k] - toFP32(smem_a))*2048)
            loadAndSplit<float4, float>(A, K, innerRowA, i * rowStrideA, innerColA, As, dAs, BK);
        }
        for (int i = 0; i < loadIterationsB; i++)
        {
            // smem_b = toFP16(mem_b[k]), smem_db = toFP16((mem_b[k] - toFP32(smem_b))*2048)
            loadAndSplit<float4, float>(B, N, innerRowB, i * rowStrideB, innerColB, Bs, dBs, BN);
        }

        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // start of data belonging to respective warp
        half *warpAs = &As[warpRow * WM * BK];
        half *warpBs = &Bs[warpCol * WN];
        half *warpdAs = &dAs[warpRow * WM * BK];
        half *warpdBs = &dBs[warpCol * WN];
        
        // calculate mmul
        for (int tileRow = 0; tileRow < N_WMMA_ROWS_PER_WARP; tileRow++)
        {
            for (int tileCol = 0; tileCol < N_WMMA_COLS_PER_WARP; tileCol++)
            {
                // Warning: if this is removed and the compiler unrolls this loop, register usage for this kernel is too high
                // and it can't be launched
                #pragma unroll 1
                for (int chunk = 0; chunk < CHUNK_K; chunk++)
                {   
                    // load_matrix_sync(frag_a, smem_a)
                    wmma::load_matrix_sync(aFrag, warpAs + chunk * WMMA_K, BK);
                    // load_matrix_sync(frag_b, smem_b)
                    wmma::load_matrix_sync(bFrag, warpBs + chunk * WMMA_K * BN, BN);

                    wmma::fill_fragment(tmpFrag, 0.0f);
                    wmma::mma_sync(tmpFrag, aFrag, bFrag, tmpFrag);

                    for (int i = 0; i < cFrag[tileRow][tileCol].num_elements; i++)
                    {
                        cFrag[tileRow][tileCol].x[i] += tmpFrag.x[i];
                    }
                    
                    // load_matrix_sync(frag_da, smem_da)
                    wmma::load_matrix_sync(daFrag, warpdAs + chunk * WMMA_K, BK);
                    // load_matrix_sync(frag_db, smem_db)
                    wmma::load_matrix_sync(dbFrag, warpdBs + chunk * WMMA_K * BN, BN);
                    // matrices for error correction can be directly accumulated in the tensor core
                    wmma::mma_sync(dcFrag[tileRow][tileCol], daFrag, bFrag, dcFrag[tileRow][tileCol]);
                    wmma::mma_sync(dcFrag[tileRow][tileCol], aFrag, dbFrag, dcFrag[tileRow][tileCol]);

                }
                warpBs += WMMA_N;
                warpdBs += WMMA_N;
            }
            warpBs = &Bs[warpCol * WN];
            warpdBs = &dBs[warpCol * WN];
            warpAs += WMMA_M * BK;
            warpdAs += WMMA_M * BK;
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
            // store_matrix_sync(mem_c, frag_c)
            wmma::store_matrix_sync(warpC + tileCol * WMMA_N, cFrag[tileRow][tileCol], N, wmma::mem_row_major);
        }
        warpC += WMMA_M * N;
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int CHUNK_K,
          const int N_WARP_ROWS_PER_BLOCK,
          const int N_WARP_COLS_PER_BLOCK,
          const int N_WMMA_ROWS_PER_WARP,
          const int N_WMMA_COLS_PER_WARP>
__global__ void matmul_v2_kernel(const float *A, const float *B, float *C, size_t M, size_t K, size_t N)
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
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> daFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag[N_WMMA_COLS_PER_WARP];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> dbFrag[N_WMMA_COLS_PER_WARP];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag[N_WMMA_ROWS_PER_WARP][N_WMMA_COLS_PER_WARP];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dcFrag[N_WMMA_ROWS_PER_WARP][N_WMMA_COLS_PER_WARP];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> tmpFrag;

    for (int i = 0; i < N_WMMA_ROWS_PER_WARP; i++)
    {
        for (int j = 0; j < N_WMMA_COLS_PER_WARP; j++)
        {
            wmma::fill_fragment(cFrag[i][j], 0.0f);
            wmma::fill_fragment(dcFrag[i][j], 0.0f);
        }
    }

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

template<int version>
void matmul_Ootomo(float *A, float *B, float *C, size_t M, size_t K, size_t N) 
{
    assert((M % 16) == 0);
    assert((K % 16) == 0);
    assert((N % 16) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate gpu");
    
    size_t AElems = M * K;
    size_t BElems = K * N;
    size_t CElems = M * N;
    float *deviceAFull, *deviceBFull, *deviceCFull;
    PRINT_ON_ERROR(cudaMalloc(&deviceAFull, AElems * sizeof(float)));
    PRINT_ON_ERROR(cudaMalloc(&deviceBFull, BElems * sizeof(float)));
    PRINT_ON_ERROR(cudaMalloc(&deviceCFull, CElems * sizeof(float)));

    // these identifiers are just outside the if because otherwise, compilation does not work
    half *deviceA[2], *deviceB[2];
    // {AB, dAB, AdB, dAdB}
    float *deviceC[4];
    if constexpr(version == 0)
    {
        for(int i = 0; i < 2; i++)
        {
            PRINT_ON_ERROR(cudaMalloc(&deviceA[i], AElems * sizeof(half)));
            PRINT_ON_ERROR(cudaMalloc(&deviceB[i], BElems * sizeof(half)));
        }
        for(int i = 0; i < 4; i++)
            PRINT_ON_ERROR(cudaMalloc(&deviceC[i], CElems * sizeof(float)));
    }

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    PRINT_ON_ERROR(cudaMemcpy(deviceAFull, A, AElems * sizeof(float), cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(deviceBFull, B, BElems * sizeof(float), cudaMemcpyHostToDevice));

    if constexpr(version == 0)
    {
        PROFILE_SEGMENTS_SWITCH("split");
        int threadsPerBlock = 256;
        split_2<float, half><<<M * K / threadsPerBlock, threadsPerBlock>>>(deviceAFull, deviceA[0], deviceA[1], 2048.0f);
        PRINT_ON_ERROR(cudaGetLastError());
        split_2<float, half><<<K * N / threadsPerBlock, threadsPerBlock>>>(deviceBFull, deviceB[0], deviceB[1], 2048.0f);
        PRINT_ON_ERROR(cudaGetLastError());

        PRINT_ON_ERROR(cudaDeviceSynchronize());
    }

    PROFILE_SEGMENTS_SWITCH("matmul");
    if constexpr(version == 0)
    {
        if (M < 256 | K < 256 | N < 256)
        {
            constexpr struct matmulTemplateArgs p = getMatmulTemplateArgs<0>();
            assert(M % p.BM == 0);
            assert(N % p.BN == 0);
            assert(K % p.BK == 0);

            dim3 blocks(M / p.BM, N / p.BN);
            for (int i = 0; i < 4; i++)
            {
                matmul_v0_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                <<<blocks, p.threadsPerBlock>>>(deviceA[i%2], deviceB[i/2], deviceC[i], M, K, N);
                PRINT_ON_ERROR(cudaGetLastError());
            }
        } 
        else 
        {
            constexpr struct matmulTemplateArgs p = getMatmulTemplateArgs<1>();
            assert(M % p.BM == 0);
            assert(N % p.BN == 0);
            assert(K % p.BK == 0);

            dim3 blocks(M / p.BM, N / p.BN);
            for (int i = 0; i < 4; i++)
            {
                matmul_v0_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                <<<blocks, p.threadsPerBlock>>>(deviceA[i%2], deviceB[i/2], deviceC[i], M, K, N);
                PRINT_ON_ERROR(cudaGetLastError());
            }
        }
    } 
    else if constexpr(version == 1 || version == 2)
    {   
        if (M < 256 | K < 256 | N < 256)
        {
            constexpr struct matmulTemplateArgs p = getMatmulTemplateArgs<0>();
            assert(M % p.BM == 0);
            assert(N % p.BN == 0);
            assert(K % p.BK == 0);

            dim3 blocks(M / p.BM, N / p.BN);
            if (version == 1)
            {
                matmul_v1_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                        <<<blocks, p.threadsPerBlock>>>(deviceAFull, deviceBFull, deviceCFull, M, K, N);
            } 
            else 
            {
                matmul_v2_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                        <<<blocks, p.threadsPerBlock>>>(deviceAFull, deviceBFull, deviceCFull, M, K, N);
            }
            PRINT_ON_ERROR(cudaGetLastError());
        }
        else
        {
            constexpr struct matmulTemplateArgs p = getMatmulTemplateArgs<2>();
            assert(M % p.BM == 0);
            assert(N % p.BN == 0);
            assert(K % p.BK == 0);

            dim3 blocks(M / p.BM, N / p.BN);
            if (version == 1)
            {
                matmul_v1_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                        <<<blocks, p.threadsPerBlock>>>(deviceAFull, deviceBFull, deviceCFull, M, K, N);
            } 
            else 
            {
                matmul_v2_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                        <<<blocks, p.threadsPerBlock>>>(deviceAFull, deviceBFull, deviceCFull, M, K, N);
            }
            PRINT_ON_ERROR(cudaGetLastError());
        }
    }

    PRINT_ON_ERROR(cudaDeviceSynchronize());

    if constexpr(version == 0)
    {
        PROFILE_SEGMENTS_SWITCH("merge");
        int threadsPerBlock = 256;
        merge_2<float, float, false><<<M * N / threadsPerBlock, threadsPerBlock>>>(deviceCFull, deviceC[0], deviceC[1], deviceC[2], deviceC[3], 2048.0f);
        PRINT_ON_ERROR(cudaGetLastError());

        PRINT_ON_ERROR(cudaDeviceSynchronize());
    }

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    PRINT_ON_ERROR(cudaMemcpy(C, deviceCFull, CElems * sizeof(float), cudaMemcpyDeviceToHost));


    PROFILE_SEGMENTS_SWITCH("free");
    PRINT_ON_ERROR(cudaFree(deviceAFull));
    PRINT_ON_ERROR(cudaFree(deviceBFull));
    PRINT_ON_ERROR(cudaFree(deviceCFull));

    if constexpr(version == 0)
    {
        for(int i = 0; i < 2; i++)
        {
            PRINT_ON_ERROR(cudaFree(deviceA[i]));
            PRINT_ON_ERROR(cudaFree(deviceB[i]));
        }
        for(int i = 0; i < 4; i++)
            PRINT_ON_ERROR(cudaFree(deviceC[i]));
    }

    PROFILE_SEGMENT_FUNCTION_END();
}

/**
 * flops16:
 * 4*(2*M*K*N) (4 matmuls)
 * 
 * flops32:
 * 2*M*K flops32 + 2*K*N flops32 (splitting A and B)
 * + 5*N*M flops32 (merging with merge_cuda)
 */
flop_counts matmul_Ootomo_v0(float *A, float *B, float *C, size_t M, size_t K, size_t N)
{
    matmul_Ootomo<0>(A, B, C, M, K, N);
    flop_counts counts = {8L*M*K*N, 2L*M*K + 2L*K*N + 5L*N*M, 0L};
    return counts;
}

/**
 * flops16:
 * 3*(2*M*K*N) (3 matmuls)
 * 
 * flops32:
 * 2*M*K + 2*K*N (splitting A and B)
 * + N*M (accumulating outside tensor cores)
 * + 2*N*M (merging into C)
 * 
 * NOTE: merging/accumulation flops32 should double-checked again
 */
flop_counts matmul_Ootomo_v1(float *A, float *B, float *C, size_t M, size_t K, size_t N)
{
    matmul_Ootomo<1>(A, B, C, M, K, N);
    flop_counts counts = {6L*M*K*N, 2L*M*K + 2L*K*N + 3L*N*M, 0L};
    return counts;
}

/**
 * flops16:
 * 3*(2*M*K*N) (3 matmuls)
 * 
 * flops32:
 * 2*M*K + 2*K*N (splitting A and B)
 * + N*M (accumulating outside tensor cores)
 * + 2*N*M (merging into C)
 * 
 * NOTE: merging/accumulation flops32 should double-checked again
 */
flop_counts matmul_Ootomo_v2(float *A, float *B, float *C, size_t M, size_t K, size_t N)
{
    matmul_Ootomo<2>(A, B, C, M, K, N);
    flop_counts counts = {6L*M*K*N, 2L*M*K + 2L*K*N + 3L*N*M, 0L};
    return counts;
}


template<int version>
void matmul_Ootomo_double(double *A, double *B, double *C, size_t M, size_t K, size_t N) 
{
    assert((M % 16) == 0);
    assert((K % 16) == 0);
    assert((N % 16) == 0);

    PROFILE_FUNCTION_SEGMENT_START("allocate gpu");
    
    size_t AElems = M * K;
    size_t BElems = K * N;
    size_t CElems = M * N;
    double *deviceAFull, *deviceBFull, *deviceCFull;
    PRINT_ON_ERROR(cudaMalloc(&deviceAFull, AElems * sizeof(double)));
    PRINT_ON_ERROR(cudaMalloc(&deviceBFull, BElems * sizeof(double)));
    PRINT_ON_ERROR(cudaMalloc(&deviceCFull, CElems * sizeof(double)));

    // these identifiers are just outside the if because otherwise, compilation does not work
    float *deviceA[2], *deviceB[2];
    // {AB, dAB, AdB, dAdB}
    float *deviceC[4];
    if constexpr(version == 0)
    {
        for(int i = 0; i < 2; i++)
        {
            PRINT_ON_ERROR(cudaMalloc(&deviceA[i], AElems * sizeof(float)));
            PRINT_ON_ERROR(cudaMalloc(&deviceB[i], BElems * sizeof(float)));
        }
        for(int i = 0; i < 4; i++)
            PRINT_ON_ERROR(cudaMalloc(&deviceC[i], CElems * sizeof(float)));
    }

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    PRINT_ON_ERROR(cudaMemcpy(deviceAFull, A, AElems * sizeof(double), cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(deviceBFull, B, BElems * sizeof(double), cudaMemcpyHostToDevice));

    if constexpr(version == 0)
    {
        PROFILE_SEGMENTS_SWITCH("split");
        int threadsPerBlock = 256;
        split_2<double, float><<<M * K / threadsPerBlock, threadsPerBlock>>>(deviceAFull, deviceA[0], deviceA[1], 1 << 24);
        PRINT_ON_ERROR(cudaGetLastError());
        split_2<double, float><<<K * N / threadsPerBlock, threadsPerBlock>>>(deviceBFull, deviceB[0], deviceB[1], 1 << 24);
        PRINT_ON_ERROR(cudaGetLastError());

        PRINT_ON_ERROR(cudaDeviceSynchronize());
    }

    PROFILE_SEGMENTS_SWITCH("matmul");
    if constexpr(version == 0)
    {
        if (M < 256 | K < 256 | N < 256)
        {
            constexpr struct matmulTemplateArgs p = getMatmulTemplateArgs<0>();
            assert(M % p.BM == 0);
            assert(N % p.BN == 0);
            assert(K % p.BK == 0);

            dim3 blocks(M / p.BM, N / p.BN);
            for (int i = 0; i < 4; i++)
            {
                matmul_v2_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                    <<<blocks, p.threadsPerBlock>>>(deviceA[i%2], deviceB[i/2], deviceC[i], M, K, N);
                PRINT_ON_ERROR(cudaGetLastError());
            }
        } 
        else 
        {
            constexpr struct matmulTemplateArgs p = getMatmulTemplateArgs<1>();
            assert(M % p.BM == 0);
            assert(N % p.BN == 0);
            assert(K % p.BK == 0);

            dim3 blocks(M / p.BM, N / p.BN);
            for (int i = 0; i < 4; i++)
            {
                matmul_v2_kernel<p.BM, p.BN, p.BK, p.WM, p.WN, p.CHUNK_K, p.N_WARP_ROWS_PER_BLOCK, p.N_WARP_COLS_PER_BLOCK, p.N_WMMA_ROWS_PER_WARP, p.N_WMMA_COLS_PER_WARP>
                    <<<blocks, p.threadsPerBlock>>>(deviceA[i%2], deviceB[i/2], deviceC[i], M, K, N);
                PRINT_ON_ERROR(cudaGetLastError());
            }
        }
    } 

    PRINT_ON_ERROR(cudaDeviceSynchronize());

    if constexpr(version == 0)
    {
        PROFILE_SEGMENTS_SWITCH("merge");
        int threadsPerBlock = 256;
        merge_2<float, double, true><<<M * N / threadsPerBlock, threadsPerBlock>>>(deviceCFull, deviceC[0], deviceC[1], deviceC[2], deviceC[3], 1 << 24);
        PRINT_ON_ERROR(cudaGetLastError());

        PRINT_ON_ERROR(cudaDeviceSynchronize());
    }

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    PRINT_ON_ERROR(cudaMemcpy(C, deviceCFull, CElems * sizeof(double), cudaMemcpyDeviceToHost));


    PROFILE_SEGMENTS_SWITCH("free");
    PRINT_ON_ERROR(cudaFree(deviceAFull));
    PRINT_ON_ERROR(cudaFree(deviceBFull));
    PRINT_ON_ERROR(cudaFree(deviceCFull));

    if constexpr(version == 0)
    {
        for(int i = 0; i < 2; i++)
        {
            PRINT_ON_ERROR(cudaFree(deviceA[i]));
            PRINT_ON_ERROR(cudaFree(deviceB[i]));
        }
        for(int i = 0; i < 4; i++)
            PRINT_ON_ERROR(cudaFree(deviceC[i]));
    }

    PROFILE_SEGMENT_FUNCTION_END();
}


/**
 * flops16:
 * 4*3*(2*M*K*N) (4 fp32 matmuls each of which is 3 fp16)
 * 
 * flops32:
 * 4 *            (4 fp32 matmuls)
 * (2*M*K + 2*K*N (splitting A and B)
 * + N*M          (accumulating outside tensor cores)
 * + 2*N*M)       (merging into C)
 * 
 * flops64:
 * 2*M*K flops64 + 2*K*N flops64 (splitting A and B)
 * + 5*N*M flops64               (merging with merge_cuda)
 */
flop_counts matmul_Ootomo_double_v0(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    matmul_Ootomo_double<0>(A, B, C, M, K, N);
    flop_counts counts = {24L*M*K*N, 4L*(2L*M*K + 2L*K*N + 3L*N*M), 2L*M*K + 2L*K*N + 5L*N*M};
    return counts;
}
