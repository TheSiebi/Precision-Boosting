#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#define DivRoundUp(Value, Dividend) (((Value) + ((Dividend) - 1)) / (Dividend))

#define PRINT_ON_ERROR(expr) do{ \
    cudaError_t err = (expr); \
    if(err != cudaSuccess) \
    { printf("CUDA ERROR: %s %s at %s:%d\n", cudaGetErrorName(err), cudaGetErrorString(err), __FILE__, __LINE__); } \
    }while(false)


#ifdef NPROFILER
#define CUDA_DEVICE_SYNCHRONIZE() 
#else
#define CUDA_DEVICE_SYNCHRONIZE() PRINT_ON_ERROR(cudaDeviceSynchronize())
#endif

#endif //CUDA_UTILS_H
