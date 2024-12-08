#include "ozaki.h"
#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "../timer.h"
#include "../profiler.h"

#include "../matmul.h"

#include <cuda_fp16.h>

#define MAX_SPLITS 10

size_t ix(size_t row, size_t col, size_t rows, size_t cols)
{
    return col + row * cols;
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

std::vector<std::vector<half>> ozaki_split_to_half(const size_t m, const size_t n, double* a, const int l)
{
    // q = size(A, 2);
    // This simply means q := n
    const int q = n;

    // k = 1;
    // Keeps consistency with paper. We'll subtract one every time we index into D.
    int k = 1;

    // beta = fl(...)
    const double log2u = -11.f; // half precision
    const double beta = ceil((-log2u + log2(q)) / 2.0);

    // D{1} = zeros(size(A));
    std::vector<std::vector<half>> D = { std::vector<half>(m * n, __float2half(0.f)) };

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
        std::vector<double> w(m);
        for (size_t i = 0; i < m; ++i)
            w[i] = exp2(ceil(log2(mu[i])) + beta);

        // S = repmat(w, 1, q);
        std::vector<double> S(m * n);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                S[ix(i, j, m, n)] = w[ix(i, 0, m, 1)];

        // D{k} = fl((A + S) - S);
        // A = fl(A - D{k});
        D.resize(k, std::vector<half>(m * n));
        for (size_t ij = 0; ij < m * n; ++ij)
        {
            // Note: unclear from paper whether ((A+S)-S) computation should happen as double or half
            double intermediate = a[ij] + S[ij];
            asm volatile("" : : "r,m"(intermediate) : "memory"); // avoid compiler optimizations
            intermediate -= S[ij];
            D[k - 1][ij] = __float2half((float) intermediate);
            a[ij] -= __half2float(D[k - 1][ij]);
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
        D.resize(k, std::vector<half>(m * n));
        for (size_t ij = 0; ij < m * n; ++ij)
            D[k - 1][ij] = __float2half((float) a[ij]); // Downcasting? Paper just says D{k} = A
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

// Ozaki paper uses A [m, n] and B [n, p] matrices
template<int version>
flop_counts matmul_ozaki(double *a, double *b, double *c, size_t m, size_t n, size_t p)
{
    // Ozaki splitting modifies input matrices. Therefore, copies must be made.
    std::vector<double> a_copy(a, a + m * n);
    std::vector<double> b_copy(b, b + n * p);

    // Splitting configuration (nA, nB) influences flop-count, and must be retrieved.
    int64_t nA, nB;

    PROFILE_FUNCTION_START();
    const auto unevaluated_sum = ozaki_mul<version>(m, n, p, a_copy.data(), b_copy.data(), &nA, &nB);
    // std::cout << "nA: " << nA << "/" << MAX_SPLITS << ", ";
    // std::cout << "nB: " << nB << "/" << MAX_SPLITS << "\n";
    memset(c, 0, m * p * sizeof(double));
    for (size_t ij = 0; ij < m * p; ++ij)
        for (const auto& matrix: unevaluated_sum)
            c[ij] += matrix[ij];

    PROFILE_FUNCTION_END();

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
