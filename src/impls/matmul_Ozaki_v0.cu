#include <algorithm>
#include <vector>
#include "../timer.h"

int ix(int row, int col, int rows, int cols)
{
    return col + row * cols;
}

// Turns a rows x cols matrix into a cols x rows matrix
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
void matmul_triple_loop(const int m, const int k, const int n, const T* a, const T* b, T* c)
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            T sum = (T) 0;
            for (int l = 0; l < k; ++l)
                sum += a[ix(row, l, m, k)] * b[ix(l, col, k, n)];
            c[ix(row, col, m, n)] = sum;
        }
    }
}

/**
 * Implementation of Algorithm 3 in Ozaki paper.
 * Returns an unevaluated sum as a vector of matrices stored as vectors.
 * Uses fp32 (float) to emulate fp64 (double) precision.
 * Completely disregards sparsity criterion.
 */
std::vector<std::vector<float>> ozaki_split(const int m, const int n, double* a, const int l)
{
    // q = size(A, 2);
    // This simply means q := n
    const int q = n;

    // k = 1;
    // Keeps consistency with paper. We'll subtract one every time we index into D.
    int k = 1;

    // beta = fl(...)
    const float log2u = -24.f;
    const float beta = ceilf((-log2u + log2(q)) / 2.f);

    // D{1} = zeros(size(A));
    std::vector<std::vector<float>> D = { std::vector<float>(m * n, 0.f) };

    // while(k < l)
    while (k < l)
    {
        // mu = max(abs(A), [], 2);
        std::vector<double> mu(m, 0.0);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < q; ++j)
                mu[i] = fmax(mu[i], fabs(a[ix(i, j, m, q)]));

        // if(max(mu) == 0) -> return
        double max = 0.0;
        for (const auto mui: mu)
            max = fmax(max, mui);
        if (max == 0.0)
        {
            // printf("Early termination\n");
            return D;
        }

        // w = fl(...);
        std::vector<double> w(m);
        for (int i = 0; i < m; ++i)
            w[i] = exp2f(ceilf((float) log2f(mu[i])) + beta);

        // S = repmat(w, 1, q);
        std::vector<double> S(m * n);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                S[ix(i, j, m, n)] = w[ix(i, 0, m, 1)];

        // D{k} = fl((A + S) - S);
        // A = fl(A - D{k});
        D.resize(k, std::vector<float>(m * n));
        for (int ij = 0; ij < m * n; ++ij)
        {
            D[k - 1][ij] = ((float) a[ij] + (float) S[ij]) - (float) S[ij]; // Compile with -O0!
            a[ij] = (double) ((float) a[ij] - D[k - 1][ij]);
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
        memcpy(D[k - 1].data(), a, m * n * sizeof(double));
    }

    return D;

}


/**
 * Implementation of Algorithm 4 in Ozaki paper.
 * Returns an unevaluated sum as a vector of matrices stored as vectors.
 * Uses fp32 (float) to emulate fp64 (double) precision.
 * Completely disregards sparsity criterion.
 */
std::vector<std::vector<float>> ozaki_mul(const int m, const int n, const int p, double* a, double* b, int* nA_ptr, int* nB_ptr)
{
    // [m, n] = size(A); [n, p] = size(B);
    // Given as parameters

    // D = Split_Mat(A, inf, delta); nA = length(D);
    const auto D = ozaki_split(m, n, a, INT_MAX);
    const auto nA = D.size();
    *nA_ptr = (int) nA;

    // E = Split_Mat(BT, inf, delta); nB = length(E);
    // Do we really need to transpose B?
    // transpose(n, p, b);
    const auto E = ozaki_split(n, p, b, INT_MAX); // remove const qualifier if you do wish to transpose
    const auto nB = E.size();
    *nB_ptr = (int) nB;

    // for r = 1 : nB, E{r} = E{r}T ; end
    // again, why transpose?
    // for (auto& matrix: E)
    //     transpose(p, n, matrix.data());

    int t = 0;
    std::vector<std::vector<float>> C(nA * nB, std::vector<float>(m * p));
    for (int r = 0; r < nA; ++r)
        for (int s = 0; s < nB; ++s)
            matmul_triple_loop<float>(m, n, p, D[r].data(), E[s].data(), C[t++].data());

    return C;

}

// WARNING: data in a, b, will be modified!
// Ozaki paper uses A [m, n] and B [n, p] matrices
flop_counts matmul_Ozaki_v0(double *a, double *b, double *c, int m, int n, int p)
{
    int nA, nB;
    const auto unevaluated_sum = ozaki_mul(m, n, p, a, b, &nA, &nB);
    memset(c, 0, m * p * sizeof(double));
    for (int ij = 0; ij < m * p; ++ij)
        for (const auto& matrix: unevaluated_sum)
            c[ij] += matrix[ij];

    flop_counts counts = 
    {
        0L,
        8L + (4L*m+3L*m*n)*nA + (4L*n+3L*n*p)*nB + 2L*nA*nB*m*n*p,
        (2L*m*n+m)*nA + (2L*n*p+n)*nB + m*p*nA*nB
    };
    return counts;
}

flop_counts matmul_Ozaki_v0_sort_then_accumulate(double *a, double *b, double *c, int m, int n, int p)
{
    int nA, nB;
    const auto unevaluated_sum = ozaki_mul(m, n, p, a, b, &nA, &nB);
    memset(c, 0, m * p * sizeof(double));
    for (int ij = 0; ij < m * p; ++ij)
    {
        std::vector<float> summands(unevaluated_sum.size());
        auto it = summands.begin();
        for (const auto& matrix: unevaluated_sum)
            *(it++) = matrix[ij];
        std::sort(summands.begin(), summands.end());
        for (const auto s: summands)
            c[ij] += s;
    }

    flop_counts counts = 
    {
        0L,
        8L + (4L*m+3L*m*n)*nA + (4L*n+3L*n*p)*nB + 2L*nA*nB*m*n*p,
        (2L*m*n+m)*nA + (2L*n*p+n)*nB + m*p*nA*nB
    };
    return counts;
}