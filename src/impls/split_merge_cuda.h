#include <cuda.h>
#include <cuda_fp16.h>

/**
 * Simple kernel that splits a float matrix into two half matrices according to the Ootoma paper
 */
template<typename srcType, typename trgtType>
__global__ void split_2(srcType *A, trgtType *A0, trgtType *A1, srcType factor);

/**
 * Simple kernel that performs the merge described in the Ootomo paper including the smallest term. 
 */
template<typename srcType, typename trgtType, bool useLastTerm>
__global__ void merge_2(trgtType *C, srcType *AB, srcType *dAB, srcType *AdB, srcType *dAdB, trgtType factor);

template<typename srcType, typename trgtType, typename returnType>
__device__ returnType split_element(srcType elem);

__global__ void split_4_tree(double *A, half *dA_high, half *dA_middleUp, half *dA_middleDown, half *dA_low);