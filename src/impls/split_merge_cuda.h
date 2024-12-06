#ifndef SPLIT_MERGE_CUDA_H
#define SPLIT_MERGE_CUDA_H

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


template<int splitCount, typename trgtType>
__global__
void split_cuda_double(double *A, trgtType *ASplit, int N, double scale)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < N)
    {
        double residual = A[i];
        double factor = 1;
        #pragma unroll
        for(int j = 0; j < splitCount; j++)
        {
            trgtType mainPart = (trgtType)(residual * factor);
            ASplit[j*N+i] = mainPart;
            residual -= (double)mainPart / factor;
            factor *= scale;
        }
    }
}

template<int splitCount>
__global__
void split_cuda_double_double(double *A, double *ASplit, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < N)
    {
        double residual = A[i];
        double factor = 1;
        #pragma unroll
        for(int j = 0; j < splitCount; j++)
        {
            double mainPart = (double)(half)(residual * factor);
            ASplit[j*N+i] = mainPart;
            residual -= mainPart / factor;
            factor *= 2048.0f;
        }
    }
}

template<int splitCount, typename srcType>
__global__ 
void merge_cuda_double(srcType *CSplit, double *C, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < N)
    {
        double result = 0;
        #pragma unroll
        for(int j = 0; j < splitCount; j++)
            result += (double)CSplit[j*N+i];
        C[i] = result;
    }
}


#endif // SPLIT_MERGE_CUDA_H