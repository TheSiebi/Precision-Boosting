#include "ozaki.h"
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "../cuda_utils.h"
#include "../timer.h"
#include "../profiler.h"
#include "split_merge_cuda.h"
#include "matmul_cuda.h"

#include "../matmul.h"

#include <cuda_fp16.h>

#define MAX_SPLITS 5

size_t ix(size_t row, size_t col, size_t rows, size_t cols)
{
    return col + row * cols;
}

static bool compareByDescendingSum(const std::pair<int, int>& a, const std::pair<int, int>& b) {
    return (a.first + a.second) > (b.first + b.second);
}

// Turns a [rows, cols] matrix into a [cols, rows] matrix
template<class T>
void transpose(const size_t rows, const size_t cols, T* data)
{
    T* backup = (T*) malloc(rows * cols * sizeof(T));
    memcpy(backup, data, rows * cols * sizeof(T));

    for (size_t i = 0; i < cols; ++i)
        for (size_t j = 0; j < rows; ++j)
            data[ix(i, j, cols, rows)] = backup[ix(j, i, rows, cols)];

    free(backup);
}

template<class T>
void matmul_triple_loop(const size_t m, const size_t k, const size_t n, const T* a, const T* b, T* c)
{
    for (size_t row = 0; row < m; ++row)
    {
        for (size_t col = 0; col < n; ++col)
        {
            T sum = (T) 0;
            for (size_t l = 0; l < k; ++l)
                sum += a[ix(row, l, m, k)] * b[ix(l, col, k, n)];
            c[ix(row, col, m, n)] = sum;
        }
    }
}

std::vector<std::vector<__half>> ozaki_split_to_half(const size_t m, const size_t n, double* a, const int l)
{
    // q = size(A, 2);
    // This simply means q := n
    const int q = n;

    // k = 1;
    // Keeps consistency with paper. We'll subtract one every time we index into D.
    int k = 1;

    // beta = fl(...)
    const float log2u = -11.f; // half precision
    const float beta = ceilf((-log2u + log2f(q)) / 2.0);

    // D{1} = zeros(size(A));
    std::vector<std::vector<__half>> D = { std::vector<__half>(m * n, static_cast<__half>(0.f)) };

    // while(k < l)
    while (k < l)
    {
        // mu = max(abs(A), [], 2);
        std::vector<double> mu(m, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < q; ++j)
                mu[i] = fmax(mu[i], fabs(a[ix(i, j, m, q)]));

        // if(max(mu) == 0) -> return
        double max = 0.0;
        for (const auto mu_i: mu)
            max = fmax(max, mu_i);
        if (max == 0.0)
        {
            // printf("Early termination\n");
            return D;
        }

        // w = fl(...);
        std::vector<__half> w(m);
        for (size_t i = 0; i < m; ++i)
            w[i] = static_cast<__half>(exp2f(ceilf(log2f(static_cast<float>(mu[i]))) + beta));

        // S = repmat(w, 1, q);
        // We'll just read from w instead
        // std::vector<__half> S(m * n);
        // for (size_t i = 0; i < m; ++i)
        //     for (size_t j = 0; j < n; ++j)
        //         S[ix(i, j, m, n)] = w[ix(i, 0, m, 1)];

        // D{k} = fl((A + S) - S);
        // A = fl(A - D{k});
        D.resize(k, std::vector<__half>(m * n));
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                const size_t ij = ix(i, j, m, n);
                volatile double intermediate1 = a[ij] + static_cast<double>(w[i]);
                volatile double intermediate2 = intermediate1 - static_cast<double>(w[i]);
                const __half value = static_cast<__half>(intermediate2);
                D[k - 1][ij] = value;
                a[ij] -= static_cast<double>(value);
            }
        }


        // % Checking sparsity of D{k}
        // Omitted

        // k = k + 1;
        ++k;
    }

    // if(k == l)
    if (k == l)
    {
        // Happens if early termination criterion was not met.
        // D{k} = A;
        // printf("Early termination not reached for k = %zu = l, D.size() = %zu\n", k, D.size());
        D.resize(k, std::vector<__half>(m * n));
        for (size_t ij = 0; ij < m * n; ++ij)
            D[k - 1][ij] = static_cast<__half>(a[ij]); // Downcasting? Paper just says D{k} = A
    }

    return D;
}

/**
 * Implementation of Algorithm 3 in Ozaki paper.
 * Returns an unevaluated sum as a vector of matrices stored as vectors.
 * Uses fp32 (float) to emulate fp64 (double) precision.
 * Completely disregards sparsity criterion.
 */
std::vector<std::vector<float>> ozaki_split_to_float(const size_t m, const size_t n, double* a, const int l)
{
    // q = size(A, 2);
    // This simply means q := n
    const int q = n;

    // k = 1;
    // Keeps consistency with paper. We'll subtract one every time we index into D.
    int k = 1;

    // beta = fl(...)
    const float log2u = -24.f;
    const float beta = ceilf((-log2u + log2f(q)) / 2.f);

    // D{1} = zeros(size(A));
    std::vector<std::vector<float>> D = { std::vector<float>(m * n, 0.f) };

    // while(k < l)
    while (k < l)
    {
        // mu = max(abs(A), [], 2);
        std::vector<double> mu(m, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < q; ++j)
                mu[i] = fmax(mu[i], fabs(a[ix(i, j, m, q)]));

        // if(max(mu) == 0) -> return
        double max = 0.0;
        for (const auto mu_i: mu)
            max = fmax(max, mu_i);
        if (max == 0.0)
        {
            // printf("Early termination\n");
            return D;
        }

        // w = fl(...);
        std::vector<float> w(m);
        for (size_t i = 0; i < m; ++i)
            w[i] = exp2f(ceilf((float) log2f(mu[i])) + beta);

        // S = repmat(w, 1, q);
        std::vector<float> S(m * n);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                S[ix(i, j, m, n)] = w[ix(i, 0, m, 1)];

        // D{k} = fl((A + S) - S);
        // A = fl(A - D{k});
        D.resize(k, std::vector<float>(m * n));
        for (size_t ij = 0; ij < m * n; ++ij)
        {
            D[k - 1][ij] = a[ij] + S[ij];
            D[k - 1][ij] -= S[ij];
            a[ij] -= D[k - 1][ij];
        }

        // % Checking sparsity of D{k}
        // Omitted

        // k = k + 1;
        ++k;
    }

    // if(k == l)
    if (k == l)
    {
        // Happens if early termination criterion was not met.
        // D{k} = A;
        // printf("Early termination not reached for k = %zu = l, D.size() = %zu\n", k, D.size());
        D.resize(k, std::vector<float>(m * n));
        for (size_t ij = 0; ij < m * n; ++ij)
            D[k - 1][ij] = (float) a[ij]; // Downcasting? Paper just says D{k} = A
    }

    return D;

}

/**
 * Implementation of Algorithm 4 in Ozaki paper.
 * Returns an unevaluated sum as a vector of matrices stored as vectors.
 * Completely disregards sparsity criterion.
 */
template<int version>
std::vector<std::vector<float>> ozaki_mul(const size_t m, const size_t n, const size_t p, double* a, double* b, int64_t* nA_ptr, int64_t* nB_ptr)
{
    // [m, n] = size(A); [n, p] = size(B);
    // Given as parameters

    if constexpr (version == 0 || version == 1)
    {
        PROFILE_SEGMENT_START("split");
        // D = Split_Mat(A, inf, delta); nA = length(D);
        auto D = ozaki_split_to_float(m, n, a, MAX_SPLITS);
        const auto nA = D.size();
        *nA_ptr = (int) nA;

        // E = Split_Mat(BT, inf, delta); nB = length(E);
        // Do we really need to transpose B?
        // transpose(n, p, b);
        auto E = ozaki_split_to_float(n, p, b, MAX_SPLITS);
        const auto nB = E.size();
        *nB_ptr = (int) nB;

        // for r = 1 : nB, E{r} = E{r}T ; end
        // again, why transpose?
        // for (auto& matrix: E)
        //     transpose(p, n, matrix.data());
        PROFILE_SEGMENTS_SWITCH("matmul");

        int t = 0;
        std::vector<std::vector<float>> C(nA * nB, std::vector<float>(m * p));
        for (int r = 0; r < nA; ++r)
        {
            for (int s = 0; s < nB; ++s)
            {
                if constexpr (version == 0)
                    matmul_triple_loop<float>(m, n, p, D[r].data(), E[s].data(), C[t++].data());
                else if constexpr (version == 1)
                    matmul_cuda<float, float, 1, false>(D[r].data(), E[s].data(), C[t++].data(), m, n, p);
                else
                    throw std::runtime_error(std::string("unimplemented ozaki version " + version));
            }
        }
        PROFILE_SEGMENT_END();
        return C;
    }

    else if constexpr (version == 2)
    {
        // D = Split_Mat(A, inf, delta); nA = length(D);
        auto D = ozaki_split_to_half(m, n, a, MAX_SPLITS);
        const auto nA = D.size();
        *nA_ptr = (int) nA;

        // E = Split_Mat(BT, inf, delta); nB = length(E);
        auto E = ozaki_split_to_half(n, p, b, MAX_SPLITS);
        const auto nB = E.size();
        *nB_ptr = (int) nB;

        int t = 0;
        std::vector<std::vector<float>> C(nA * nB, std::vector<float>(m * p));
        for (int r = 0; r < nA; ++r)
            for (int s = 0; s < nB; ++s)
                matmul_cuda<half, float, 3, true>(D[r].data(), E[s].data(), C[t++].data(), m, n, p);

        return C;
    }
    else
        throw std::runtime_error(std::string("unimplemented ozaki version " + version));

}

template<int splitCount>
void ozaki_split_to_float_fixed(double* A, float* ASplit, const size_t M, const size_t N)
{
    // beta = fl(...)
    const float beta = ceilf((log2f(N) + 24) / 2.f);

    for(int k = 0; k < splitCount-1; k++)
    {
        for(size_t i = 0; i < M; i++)
        {
            double mu = 0;
            for(size_t j = 0; j < N; j++)
                mu = std::max(mu, std::abs(A[i*N+j]));
            float w = exp2f(ceilf(log2f(mu)) + beta);

            for(size_t j = 0; j < N; j++)
            {
                size_t ij = i*N+j;
                float value = (float)(A[ij] + (double)w) - w;
                ASplit[k*M*N + ij] = value;
                A[ij] -= (double)value;
            }
        }
    }

    for (size_t ij = 0; ij < M * N; ++ij)
        ASplit[(splitCount-1)*M*N + ij] = (float) A[ij];
}

// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
#define TRANSPOSE_TILE_DIM 32
#define TRANSPOSE_BLOCK_ROWS 8
template<typename T>
__global__ void transpose_matrix(T *oData, T* iData, size_t M, size_t K)
{
    __shared__ T tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM];

    size_t x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    size_t y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = iData[(y + j)*K + x];

    __syncthreads();

    x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
        oData[(y + j)*M + x] = tile[threadIdx.x][threadIdx.y + j];
}

template <typename T>
void transposeMatrix(T *A, T *A_T, size_t M, size_t K)
{
    size_t size = M * K;
    T *d_A, *d_A_T;
    PRINT_ON_ERROR(cudaMalloc(&d_A, size * sizeof(T)));
    PRINT_ON_ERROR(cudaMalloc(&d_A_T, size * sizeof(T)));

    PRINT_ON_ERROR(cudaMemcpy(d_A, A, size * sizeof(T), cudaMemcpyHostToDevice));

    dim3 blockDim(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
    dim3 gridDim(K / TRANSPOSE_TILE_DIM, M / TRANSPOSE_TILE_DIM);

    transpose_matrix<<<gridDim, blockDim>>>(d_A_T, d_A, M, K);
    PRINT_ON_ERROR(cudaGetLastError());
    PRINT_ON_ERROR(cudaDeviceSynchronize());

    PRINT_ON_ERROR(cudaMemcpy(A_T, d_A_T, size * sizeof(T), cudaMemcpyDeviceToHost));

    PRINT_ON_ERROR(cudaFree(d_A));
    PRINT_ON_ERROR(cudaFree(d_A_T));
}

template void transposeMatrix<float>(float *, float *, size_t, size_t); 

template<int splitCount>
static __global__
void ozaki_split_to_half_fixed_cuda(double* A, half* ASplit, const size_t M, const size_t N, const half beta)
{
    const int warpIndex = threadIdx.x;
    const int scalar_AOffset = blockIdx.x * N;

    double mu = 0;
    for(size_t j = 0; j < N; j+=WARP_SIZE)
        mu = max(mu, abs(A[scalar_AOffset+j+warpIndex]));

    for(int k = 0; k < splitCount-1; k++)
    {
        //get max mu accross all warps
        for(int i = 16; i >= 1; i/=2)
            mu = max(mu, __shfl_xor_sync(0xffffffff, mu, i, 32));
        half w = hexp2(hceil(hlog2((half)mu)) + beta);

        mu = 0;
        for(size_t j = 0; j < N; j+=WARP_SIZE)
        {
            size_t ij = scalar_AOffset+j+warpIndex;
            half value = (half)(A[ij] + (double)w) - w;
            ASplit[k*M*N + ij] = value;
            double newA = A[ij] - (double)value;
            A[ij] = newA;
            mu = max(mu, abs(newA));
        }
    }

    for (size_t j = 0; j < N; j+=WARP_SIZE)
    {
        size_t ij = scalar_AOffset+j+warpIndex;
        ASplit[(splitCount-1)*M*N + ij] = (half) A[ij];
    }
}

template<int splitCount>
static __global__
void ozaki_split_to_float_fixed_cuda(double* A, float* ASplit, const size_t M, const size_t N, const float beta)
{
    const int warpIndex = threadIdx.x;
    const int scalar_AOffset = blockIdx.x * N;

    double mu = 0;
    for(size_t j = 0; j < N; j+=WARP_SIZE)
        mu = max(mu, abs(A[scalar_AOffset+j+warpIndex]));

    for(int k = 0; k < splitCount-1; k++)
    {
        //get max mu accross all warps
        for(int i = 16; i >= 1; i/=2)
            mu = max(mu, __shfl_xor_sync(0xffffffff, mu, i, 32));
        float w = exp2f(ceilf(log2f(mu)) + beta);

        mu = 0;
        for(size_t j = 0; j < N; j+=WARP_SIZE)
        {
            size_t ij = scalar_AOffset+j+warpIndex;
            float value = (float)(A[ij] + (double)w) - w;
            ASplit[k*M*N + ij] = value;
            double newA = A[ij] - (double)value;
            A[ij] = newA;
            mu = max(mu, abs(newA));
        }
    }

    for (size_t j = 0; j < N; j+=WARP_SIZE)
    {
        size_t ij = scalar_AOffset+j+warpIndex;
        ASplit[(splitCount-1)*M*N + ij] = (float) A[ij];
    }
}


// Ozaki paper uses A [m, n] and B [n, p] matrices
template<int version>
flop_counts matmul_ozaki(double *a, double *b, double *c, size_t m, size_t n, size_t p)
{
    PROFILE_FUNCTION_START();
    // Ozaki splitting modifies input matrices. Therefore, copies must be made.
    std::vector<double> a_copy(a, a + m * n);
    std::vector<double> b_copy(b, b + n * p);

    // Splitting configuration (nA, nB) influences flop-count, and must be retrieved.
    int64_t nA, nB;

    const auto unevaluated_sum = ozaki_mul<version>(m, n, p, a_copy.data(), b_copy.data(), &nA, &nB);
    // std::cout << "nA: " << nA << "/" << MAX_SPLITS << ", ";
    // std::cout << "nB: " << nB << "/" << MAX_SPLITS << "\n";
    PROFILE_SEGMENT_START("accumulate");
    memset(c, 0, m * p * sizeof(double));
    for (const auto& matrix: unevaluated_sum)
        for (size_t ij = 0; ij < m * p; ++ij)
            c[ij] += matrix[ij];
    
    PROFILE_SEGMENT_FUNCTION_END();

    flop_counts counts =
    {
        0L,
        8L + (4L*m+3L*m*n)*nA + (4L*n+3L*n*p)*nB + 2L*nA*nB*m*n*p,
        (2L*m*n+m)*nA + (2L*n*p+n)*nB + m*p*nA*nB
    };
    return counts;
}
template flop_counts matmul_ozaki<0>(double *a, double *b, double *c, size_t m, size_t n, size_t p);
template flop_counts matmul_ozaki<1>(double *a, double *b, double *c, size_t m, size_t n, size_t p);
template flop_counts matmul_ozaki<2>(double *a, double *b, double *c, size_t m, size_t n, size_t p);

template<int splitCount, int mergeCount, typename mulInputType>
flop_counts matmul_ozaki_optimized(double *A, double *B, double *C, size_t M, size_t K, size_t N,
                                   std::pair<int, int> mergePattern[mergeCount], bool transpose)
{
    size_t ASizeF = M * K * sizeof(mulInputType);
    size_t ASize = M * K * sizeof(double);
    size_t BSizeF = K * N * sizeof(mulInputType);
    size_t BSize = K * N * sizeof(double);
    size_t CSizeF = M * N * sizeof(float);
    size_t CSize = M * N * sizeof(double);

    PROFILE_FUNCTION_SEGMENT_START("allocate gpu");

    mulInputType *deviceA, *deviceB, *deviceBTransposed;
    float *deviceC;
    double *deviceCMerged;
    double *deviceAFull, *deviceBFull, *deviceBFullTransposed;

    PRINT_ON_ERROR(cudaMalloc(&deviceA, ASizeF * splitCount));
    PRINT_ON_ERROR(cudaMalloc(&deviceB, BSizeF * splitCount));
    if (transpose)
        PRINT_ON_ERROR(cudaMalloc(&deviceBTransposed, BSizeF * splitCount));
    PRINT_ON_ERROR(cudaMalloc(&deviceC, CSizeF * mergeCount));
    PRINT_ON_ERROR(cudaMalloc(&deviceCMerged, CSize));

    PRINT_ON_ERROR(cudaMalloc(&deviceAFull, ASize));
    PRINT_ON_ERROR(cudaMalloc(&deviceBFull, BSize));
    if (transpose)
        PRINT_ON_ERROR(cudaMalloc(&deviceBFullTransposed, BSize));

    PROFILE_SEGMENTS_SWITCH("memcpy host2device");

    PRINT_ON_ERROR(cudaMemcpy(deviceAFull, A, ASize, cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(deviceBFull, B, BSize, cudaMemcpyHostToDevice));
    CUDA_DEVICE_SYNCHRONIZE();

    PROFILE_SEGMENTS_SWITCH("split");
    if constexpr(std::is_same<half, mulInputType>::value)
    {
        const half beta = ceilf((log2f(K) + 11) / 2.f);
        ozaki_split_to_half_fixed_cuda<splitCount><<<M, 32>>>(deviceAFull, deviceA, M, K, beta);

        if (transpose)
        {
            dim3 blockDim(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
            dim3 gridDim(N / TRANSPOSE_TILE_DIM, K / TRANSPOSE_TILE_DIM);
            transpose_matrix<<<gridDim, blockDim>>>(deviceBFullTransposed, deviceBFull, K, N);
            ozaki_split_to_half_fixed_cuda<splitCount><<<N, 32>>>(deviceBFullTransposed, deviceBTransposed, N, K, beta);

            dim3 gridDim1(K / TRANSPOSE_TILE_DIM, N / TRANSPOSE_TILE_DIM);
            for (int i = 0; i < splitCount; i++)
                transpose_matrix<<<gridDim1, blockDim>>>(deviceB + i * BSizeF, deviceBTransposed + i * BSizeF, N, K);
        }
        else 
            ozaki_split_to_half_fixed_cuda<splitCount><<<K, 32>>>(deviceBFull, deviceB, K, N, beta);
    }
    else
    {
        const float beta = ceilf((log2f(K) + 24) / 2.f);
        ozaki_split_to_float_fixed_cuda<splitCount><<<M, 32>>>(deviceAFull, deviceA, M, K, beta);
        
        if (transpose)
        {
            dim3 blockDim(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
            dim3 gridDim(N / TRANSPOSE_TILE_DIM, K / TRANSPOSE_TILE_DIM);
            transpose_matrix<<<gridDim, blockDim>>>(deviceBFullTransposed, deviceBFull, K, N);
            ozaki_split_to_float_fixed_cuda<splitCount><<<N, 32>>>(deviceBFullTransposed, deviceBTransposed, N, K, beta);

            dim3 gridDim1(K / TRANSPOSE_TILE_DIM, N / TRANSPOSE_TILE_DIM);
            for (int i = 0; i < splitCount; i++)
                transpose_matrix<<<gridDim1, blockDim>>>(deviceB + i * BSizeF, deviceBTransposed + i * BSizeF, N, K);
        }
        else 
            ozaki_split_to_float_fixed_cuda<splitCount><<<K, 32>>>(deviceBFull, deviceB, K, N, beta);
    }
    CUDA_DEVICE_SYNCHRONIZE();

    PROFILE_SEGMENTS_SWITCH("matmul");

#if 1
    cudaStream_t streams[mergeCount];
    for(int i = 0; i < mergeCount; i++)
        PRINT_ON_ERROR(cudaStreamCreate(&streams[i]));
    for(int i = 0; i < mergeCount; i++)
    {
        size_t aIndex = mergePattern[i].first * M * K;
        size_t bIndex = mergePattern[i].second * K * N;
        size_t cIndex = i * M * N;
        matmulCUDACoresStream<mulInputType, float, float, 1>(&deviceA[aIndex], &deviceB[bIndex], &deviceC[cIndex], M, K, N, streams[i]);
    }
#else
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed. %s: %s\n",
                cublasGetStatusName(status), cublasGetStatusString(status));
        flop_counts counts = {0L, 0L, 0L};
        return counts;
    }
    const float cublas_alpha = 1.0f;
    const float cublas_beta = 0.0f;
#if 0
    for(int i = 0; i < mergeCount; i++)
    {
        size_t aIndex = mergePattern[i].first * M * K;
        size_t bIndex = mergePattern[i].second * K * N;
        size_t cIndex = i * M * N;
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &cublas_alpha,
                             &deviceB[bIndex], N,
                             &deviceA[aIndex], K,
                             &cublas_beta,
                             &deviceC[cIndex], N);
    }
#else
    float *APointers[mergeCount];
    float *BPointers[mergeCount];
    float *CPointers[mergeCount];
    for(int i = 0; i < mergeCount; i++)
    {
        size_t aIndex = mergePattern[i].first * M * K;
        size_t bIndex = mergePattern[i].second * K * N;
        size_t cIndex = i * M * N;
        APointers[i] = &deviceA[aIndex];
        BPointers[i] = &deviceB[bIndex];
        CPointers[i] = &deviceC[cIndex];
    }
    float **device_APointers, **device_BPointers, **device_CPointers;
    PRINT_ON_ERROR(cudaMalloc(&device_APointers, sizeof(float*) * mergeCount));
    PRINT_ON_ERROR(cudaMalloc(&device_BPointers, sizeof(float*) * mergeCount));
    PRINT_ON_ERROR(cudaMalloc(&device_CPointers, sizeof(float*) * mergeCount));

    PRINT_ON_ERROR(cudaMemcpy(device_APointers, APointers, sizeof(float*) * mergeCount, cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(device_BPointers, BPointers, sizeof(float*) * mergeCount, cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(device_CPointers, CPointers, sizeof(float*) * mergeCount, cudaMemcpyHostToDevice));

    status = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K,
                                &cublas_alpha,
                                device_BPointers, N,
                                device_APointers, K,
                                &cublas_beta,
                                device_CPointers, N,
                                mergeCount);

#endif
#endif

    CUDA_DEVICE_SYNCHRONIZE();

    PROFILE_SEGMENTS_SWITCH("merge");
    merge_n_cuda<mergeCount, float, double><<<DivRoundUp(M*N, 256), 256>>>(deviceC, deviceCMerged, M*N);
    PRINT_ON_ERROR(cudaGetLastError());
    CUDA_DEVICE_SYNCHRONIZE();

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    PRINT_ON_ERROR(cudaMemcpy(C, deviceCMerged, CSize, cudaMemcpyDeviceToHost));

    PROFILE_SEGMENTS_SWITCH("free");

    PRINT_ON_ERROR(cudaFree(deviceA));
    PRINT_ON_ERROR(cudaFree(deviceB));
    PRINT_ON_ERROR(cudaFree(deviceC));
    PRINT_ON_ERROR(cudaFree(deviceCMerged));
    PRINT_ON_ERROR(cudaFree(deviceAFull));
    PRINT_ON_ERROR(cudaFree(deviceBFull));
    if (transpose)
    {
        PRINT_ON_ERROR(cudaFree(deviceBTransposed));
        PRINT_ON_ERROR(cudaFree(deviceBFullTransposed));
    }

#if 1
    for(int i = 0; i < mergeCount; i++)
        PRINT_ON_ERROR(cudaStreamDestroy(streams[i]));
#endif

    PROFILE_SEGMENT_FUNCTION_END();

    int64_t nA = splitCount;
    int64_t nB = splitCount;
    flop_counts counts =
    {
        0L,
        8L + (4L*M+3L*M*K)*nA + (4L*K+3L*K*N)*nB + 2L*nA*nB*M*K*N,
        (2L*M*K+M)*nA + (2L*K*N+K)*nB + M*N*nA*nB
    };
    return counts;
}

template<>
flop_counts matmul_ozaki<3>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    constexpr int splitCount = 4;
    constexpr int mergeCount = splitCount * splitCount;
    std::pair<int, int> merges[mergeCount];
    for(int i = 0; i < mergeCount; i++)
        merges[i] = {i/splitCount, i%splitCount};
    std::sort(std::begin(merges), std::end(merges), compareByDescendingSum);
    return matmul_ozaki_optimized<splitCount, mergeCount, float>(A, B, C, M, K, N, merges, false);
}

template<>
flop_counts matmul_ozaki<4>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    constexpr int splitCount = 5;
    constexpr int mergeCount = splitCount * splitCount;
    std::pair<int, int> merges[mergeCount];
    for(int i = 0; i < mergeCount; i++)
        merges[i] = {i/splitCount, i%splitCount};
    std::sort(std::begin(merges), std::end(merges), compareByDescendingSum);
    return matmul_ozaki_optimized<splitCount, mergeCount, float>(A, B, C, M, K, N, merges, false);
}

template<>
flop_counts matmul_ozaki<5>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    constexpr int splitCount = 6;
    constexpr int mergeCount = 36;
    constexpr int splitCountSq = splitCount * splitCount;
    std::pair<int, int> merges[splitCountSq];
    for(int i = 0; i < splitCountSq; i++)
        merges[i] = {i/splitCount, i%splitCount};
    std::sort(std::begin(merges), std::end(merges), compareByDescendingSum);
    return matmul_ozaki_optimized<splitCount, mergeCount, float>(A, B, C, M, K, N, &merges[splitCountSq - mergeCount], false);
}

template<>
flop_counts matmul_ozaki<6>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    constexpr int splitCount = 4;
    constexpr int mergeCount = 10;
    constexpr int splitCountSq = splitCount * splitCount;
    std::pair<int, int> merges[splitCountSq];
    for(int i = 0; i < splitCountSq; i++)
        merges[i] = {i/splitCount, i%splitCount};
    std::sort(std::begin(merges), std::end(merges), compareByDescendingSum);
    return matmul_ozaki_optimized<splitCount, mergeCount, float>(A, B, C, M, K, N, &merges[splitCountSq - mergeCount], false);
}


template<>
flop_counts matmul_ozaki<7>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    constexpr int splitCount = 5;
    constexpr int mergeCount = 20;
    constexpr int splitCountSq = splitCount * splitCount;
    std::pair<int, int> merges[splitCountSq];
    for(int i = 0; i < splitCountSq; i++)
        merges[i] = {i/splitCount, i%splitCount};
    std::sort(std::begin(merges), std::end(merges), compareByDescendingSum);
    return matmul_ozaki_optimized<splitCount, mergeCount, float>(A, B, C, M, K, N, &merges[splitCountSq - mergeCount], false);
}

template<>
flop_counts matmul_ozaki<8>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    constexpr int splitCount = 6;
    constexpr int mergeCount = 25;
    constexpr int splitCountSq = splitCount * splitCount;
    std::pair<int, int> merges[splitCountSq];
    for(int i = 0; i < splitCountSq; i++)
        merges[i] = {i/splitCount, i%splitCount};
    std::sort(std::begin(merges), std::end(merges), compareByDescendingSum);
    return matmul_ozaki_optimized<splitCount, mergeCount, float>(A, B, C, M, K, N, &merges[splitCountSq - mergeCount], false);
}

template<>
flop_counts matmul_ozaki<9>(double *A, double *B, double *C, size_t M, size_t K, size_t N)
{
    constexpr int splitCount = 6;
    constexpr int mergeCount = 36;
    constexpr int splitCountSq = splitCount * splitCount;
    std::pair<int, int> merges[splitCountSq];
    for(int i = 0; i < splitCountSq; i++)
        merges[i] = {i/splitCount, i%splitCount};
    std::sort(std::begin(merges), std::end(merges), compareByDescendingSum);
    return matmul_ozaki_optimized<splitCount, mergeCount, half>(A, B, C, M, K, N, &merges[splitCountSq - mergeCount], false);
}