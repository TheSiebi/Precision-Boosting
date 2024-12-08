/**
 * Requires a CUDA compiler
 */

#include <cuda_fp16.h>

#include "../matmul.h"
#include "ozaki.h"

#include <array>
#include <vector>

void test_ozaki_split_correctness(LCG* rng, const double epsilon, const size_t max_splits)
{
    const std::array<size_t, 7> rows_sizes = { 2, 4, 8, 10, 256, 1024, 2048 };
    const std::array<size_t, 5> cols_sizes = { 5, 16, 256, 1024, 4096 };
    for (const size_t rows: rows_sizes)
    {
        for (const size_t cols: cols_sizes)
        {
            const size_t size = rows * cols;
            std::cout << "Testing " << rows << "x" << cols << "=" << size << " split.\n";
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
            double max_err = 0.0;
            for (size_t ij = 0; ij < size; ++ij)
            {
                const double abs_err = fabs(matrix[ij] - backup[ij]);
                if (abs_err > epsilon)
                {
                    std::cout
                        << "\033[31m" << "[FAILURE] \033[0m (split to float)\n"
                        << "\tOccured at row " << (ij / cols) << ", col " << (ij % cols) << "\n"
                        << "\tExpected: " << backup[ij] << " Actual: " << matrix[ij] << "\n"
                        << "\tAbsolute error: \033[33m" << abs_err << "\033[0m\n";
                    return;
                }
                max_err = fmax(max_err, abs_err);
            }
            std::cout << "Max err (float): " << max_err << "\n";

            // Test split to half
            memcpy(matrix, backup, size * sizeof(double));
            const auto split_half_matrices = ozaki_split_to_half(rows, cols, matrix, max_splits);
            memset(matrix, 0, size * sizeof(double));
            for (const auto& m: split_half_matrices)
                for (size_t ij = 0; ij < size; ++ij)
                    matrix[ij] += __half2float(m[ij]);

            // Calculate error
            max_err = 0.0;
            for (size_t ij = 0; ij < size; ++ij)
            {
                const double abs_err = fabs(matrix[ij] - backup[ij]);
                if (abs_err > epsilon)
                {
                    std::cout
                        << "\033[31m" << "[FAILURE] \033[0m (split to half)\n"
                        << "\tOccured at row " << (ij / cols) << ", col " << (ij % cols) << "\n"
                        << "\tExpected: " << backup[ij] << " Actual: " << matrix[ij] << "\n"
                        << "\tAbsolute error: \033[33m" << abs_err << "\033[0m\n";
                    return;
                }
                max_err = fmax(max_err, abs_err);
            }
            std::cout << "Max err (half): " << max_err << "\n";

            delete[] matrix;
            delete[] backup;
        }
    }

    std::cout
        << "\033[32m" << "[SUCCESS]" << "\033[0m Ozaki splits\n"; // Green text
}
