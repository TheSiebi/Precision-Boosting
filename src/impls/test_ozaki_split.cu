/**
 * Requires a CUDA compiler
 */

#include <cuda_fp16.h>

#include "../matmul.h"
#include "ozaki.h"

#include <cmath>
#include <cstdint>

#include <array>
#include <iomanip>
#include <string>
#include <vector>

std::string binary_representation(const double d_)
{
    int64_t d;
    memcpy(&d, &d_, 8);
    std::string res(64, '0');
    for (size_t i = 0; i < 64; ++i)
        if ((d >> (63 - i)) & 0b1)
            res[i] = '1';
    return res;
}

void bitwise_comparison(const double expected, const double actual)
{
    std::cout << "\n          s" << std::string(11, 'e') << std::string(52, 'm') << "\n";
    std::cout << std::setw(10) << "Expected: " << binary_representation(expected) << "\n";
    std::cout << std::setw(10) << "Actual: " << binary_representation(actual) << "\n";
}

void test_ozaki_split_correctness(LCG* rng, const double epsilon, const size_t max_splits, const bool verbose)
{
    const std::array<size_t, 1> rows_sizes = { 3 };
    const std::array<size_t, 1> cols_sizes = { 4 };
    double float_max_err = 0.0;
    double half_max_err = 0.0;
    std::cout << std::fixed << std::setprecision(17);
    for (const size_t rows: rows_sizes)
    {
        for (const size_t cols: cols_sizes)
        {
            const size_t size = rows * cols;
            if (verbose)
                std::cout << "Testing " << rows << "x" << cols << " matrix split.\n";
            double* matrix = new double[size];
            double* backup = new double[size];
            gen_urand<double>(rng, matrix, size);
            memcpy(backup, matrix, size * sizeof(double));

            // Test split to float
            const auto split_float_matrices = ozaki_split_to_float(rows, cols, matrix, max_splits);
            memset(matrix, 0, size * sizeof(double));
            for (const auto& m: split_float_matrices)
                for (size_t ij = 0; ij < size; ++ij)
                    matrix[ij] += m[ij];

            // Calculate error
            for (size_t ij = 0; ij < size; ++ij)
            {
                const double abs_err = fabs(matrix[ij] - backup[ij]);
                if (abs_err > epsilon || std::isnan(matrix[ij]))
                {
                    std::cout
                        << "\033[31m" << "[FAILURE] \033[0m (split to float)\n"
                        << "\tOccured at row " << (ij / cols) << ", col " << (ij % cols) << "\n"
                        << "\tExpected: " << backup[ij] << " Actual: " << matrix[ij] << "\n"
                        << "\tAbsolute error: \033[33m" << abs_err << "\033[0m\n";
                    bitwise_comparison(backup[ij], matrix[ij]);
                    return;
                }
                float_max_err = fmax(float_max_err, abs_err);
            }
            if (verbose)
                std::cout << "Max err after " << split_float_matrices.size() << " splits (float): " << float_max_err << "\n";

            // Test split to half
            memcpy(matrix, backup, size * sizeof(double));
            const auto split_half_matrices = ozaki_split_to_half(rows, cols, matrix, max_splits);
            memset(matrix, 0, size * sizeof(double));
            for (const auto& m: split_half_matrices)
                for (size_t ij = 0; ij < size; ++ij)
                    matrix[ij] += static_cast<double>(m[ij]);

            // Calculate error
            for (size_t ij = 0; ij < size; ++ij)
            {
                const double abs_err = fabs(matrix[ij] - backup[ij]);
                if (abs_err > epsilon || std::isnan(matrix[ij]))
                {
                    std::cout
                        << "\033[31m" << "[FAILURE] \033[0m (split to half)\n"
                        << "\tOccured at row " << (ij / cols) << ", col " << (ij % cols) << "\n"
                        << "\tExpected: " << backup[ij] << " Actual: " << matrix[ij] << "\n"
                        << "\tAbsolute error: \033[33m" << abs_err << "\033[0m\n";
                    bitwise_comparison(backup[ij], matrix[ij]);
                    // return;
                }
                half_max_err = fmax(half_max_err, abs_err);
            }
            if (verbose)
                std::cout << "Max err after " << split_half_matrices.size() << " splits (half): " << half_max_err << "\n";

            delete[] matrix;
            delete[] backup;
        }
    }

    std::cout
        << "\033[32m" << "[SUCCESS]" << "\033[0m  Ozaki splits. " // Green text
        << "Max absolute errors: " << float_max_err << " (float), " << half_max_err << " (half).\n";
}
