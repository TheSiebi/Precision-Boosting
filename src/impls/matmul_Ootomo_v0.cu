#include <assert.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../matmul.h"
#include "../profiler.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int N_WARP_ROWS_PER_BLOCK,
          const int N_WARP_COLS_PER_BLOCK,
          const int N_WMMA_ROWS_PER_WARP,
          const int N_WMMA_COLS_PER_WARP>
__global__ void matmul_v0_kernel(const float *A, const float *B, float *C, int M, int K, int N)
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

    const int warpID = threadIdx.x / WARP_SIZE;
    const int laneID = threadIdx.x % WARP_SIZE;

    // Calculate indices that this thread will load from GMEM to SMEM
    // Loads are vectorized and each thread will load 4 elements into SMEM
    // Note that for coalescing, it's important that consecutive threadIDs
    // access consecutive memory addresses
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);

    // Loop over all block tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // populate SMEM cache
        // transpose A while loading, i.e. save it in col-major order
    }

}

void matmul_Oootomo_v0(float *A, float *B, float *C, int M, int K, int N) 
{
    assert((M % 16) == 0);
    assert((K % 16) == 0);
    assert((N % 16) == 0);

    PROFILE_SEGMENTS_SWITCH("allocate gpu");
    
    int ASize = M * K * sizeof(float);
    int BSize = K * N * sizeof(float);
    int CSize = M * N * sizeof(float);
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, ASize);
    cudaMalloc(&deviceB, BSize);
    cudaMalloc(&deviceC, CSize);

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    cudaMemcpy(deviceA, A, ASize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, BSize, cudaMemcpyHostToDevice);

    PROFILE_SEGMENTS_SWITCH("matmul");

    constexpr int CHUNK_K = 2;
    constexpr int BM = WMMA_M * 8;
    constexpr int BN = WMMA_N * 8;
    constexpr int BK = WMMA_K * CHUNK_K;
    assert(M % BM == 0);
    assert(N % BN == 0);
    assert(K % BK == 0);
    constexpr int WM = WMMA_M * 4;
    constexpr int WN = WMMA_N * 4;
    static_assert(BM % WM == 0);
    static_assert(BN % WN == 0);
    constexpr int N_WARP_ROWS_PER_BLOCK = BM / WM;
    constexpr int N_WARP_COLS_PER_BLOCK = BN / WN;
    constexpr int N_WMMA_ROWS_PER_WARP = WM / WMMA_M;
    constexpr int N_WMMA_COLS_PER_WARP = WN / WMMA_N;
    constexpr int threadsPerBlock = N_WARP_ROWS_PER_BLOCK * N_WARP_COLS_PER_BLOCK * WARP_SIZE;
    dim3 blocks(M / BM, N / BN);
    matmul_v0_kernel<BM, BN, BK, WM, WN, N_WARP_ROWS_PER_BLOCK, N_WARP_COLS_PER_BLOCK, N_WMMA_ROWS_PER_WARP, N_WMMA_COLS_PER_WARP>
        <<<blocks, threadsPerBlock>>>(deviceA, deviceB, deviceC, M, K, N);

    cudaDeviceSynchronize();

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    cudaMemcpy(C, deviceC, CSize, cudaMemcpyDeviceToHost);


    PROFILE_SEGMENTS_SWITCH("free");
    free(deviceA);
    free(deviceB);
    free(deviceC);

    PROFILE_SEGMENT_FUNCTION_END();
}

