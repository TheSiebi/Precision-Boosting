#include "./split_merge_cuda.h"

template<typename srcType, typename trgtType>
__global__ void split_2(srcType *A, trgtType *A0, trgtType *A1, srcType factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    srcType value = A[i];
    trgtType mainPart = (trgtType)value;
    A0[i] = mainPart;
    A1[i] = (trgtType)((value - (srcType)mainPart) * factor);
}

template __global__ void split_2<float, half>(float*, half*, half*, float);
template __global__ void split_2<double, float>(double*, float*, float*, double);

template<typename srcType, typename trgtType, bool useLastTerm>
__global__ void merge_2(trgtType *C, srcType *AB, srcType *dAB, srcType *AdB, srcType *dAdB, trgtType factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr(useLastTerm)
        C[i] = (trgtType) AB[i] + ((trgtType) dAB[i] / factor + ((trgtType) AdB[i] / factor + (trgtType) dAdB[i] / (factor * factor)));
    else 
        C[i] = (trgtType) AB[i] + ((trgtType) dAB[i] + (trgtType) AdB[i]) / factor;
}

template __global__ void merge_2<float, float, true>(float*, float*, float*, float*, float*, float);
template __global__ void merge_2<float, float, false>(float*, float*, float*, float*, float*, float);
template __global__ void merge_2<float, double, true>(double*, float*, float*, float*, float*, double);

template<typename srcType, typename trgtType, typename returnType>
__device__ returnType split_element(srcType elem)
{
    returnType result;
    result.x = (trgtType)elem;
    result.y = (trgtType)((elem - (srcType)result.x));

    return result;
}

__global__ void split_4_tree(double *A, half *dA_high, half *dA_middleUp, half *dA_middleDown, half *dA_low)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double elem = A[i];
    float2 floatSplit = split_element<double, float, float2>(elem);
    half2 highSplit = split_element<float, half, half2>(floatSplit.x);
    half2 lowSplit = split_element<float, half, half2>(floatSplit.y);

    //double reconstructed = ((double)highSplit.x + (double)highSplit.y) + ((double)lowSplit.x + (double)lowSplit.y);
    //assert(fabs(reconstructed - A[i]) < 1e-9);
}



