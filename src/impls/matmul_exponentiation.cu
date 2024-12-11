#include <cuda_runtime.h>
#include <assert.h>

#include "../matmul.h"
#include "../profiler.h"
#include "../cuda_utils.h"
#include "../timer.h"

// Compute C = A^50
flop_counts matmul_exponentiation(half *h_A, half *h_B, half *h_C, size_t M, size_t K, size_t N) {
    // A always stores original matrix, B stores current A^exponent and C stores final result
    PROFILE_FUNCTION_SEGMENT_START("allocate");
    half *d_A, *d_B, *d_C;
    PRINT_ON_ERROR(cudaMalloc((void**)&d_A, M * K * sizeof(half)));
    PRINT_ON_ERROR(cudaMalloc((void**)&d_B, K * N * sizeof(half)));
    //PRINT_ON_ERROR(cudaMalloc((void**)&d_C, M * N * sizeof(half)));

    // Copy data from host to device
    PROFILE_SEGMENTS_SWITCH("memcpy host2device");
    PRINT_ON_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    PRINT_ON_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // Perform matrix exponentiation
    PROFILE_SEGMENTS_SWITCH("exponentiation");
    for (int i = 0; i < matmul_exponent; i++) {
        matmul_cuda<half, half, 5, true>(d_A, d_B, d_B, M, K, N);
    }

    CUDA_DEVICE_SYNCHRONIZE();

    PROFILE_SEGMENTS_SWITCH("memcpy device2host");
    // Copy the result back to host
    // NOTE: unusually, result is stored in (device) B
    PRINT_ON_ERROR(cudaMemcpy(h_C, d_B, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    PROFILE_SEGMENTS_SWITCH("free");
    // Free device memory
    PRINT_ON_ERROR(cudaFree(d_A));
    PRINT_ON_ERROR(cudaFree(d_B));
    //PRINT_ON_ERROR(cudaFree(d_C));
    PROFILE_SEGMENT_FUNCTION_END();

    flop_counts counts = {2L*M*K*N*matmul_exponent, 0L, 0L};
    return counts;
}
