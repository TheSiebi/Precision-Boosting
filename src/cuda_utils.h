
#define DivRoundUp(Value, Dividend) (((Value) + ((Dividend) - 1)) / (Dividend))

#define PRINT_ON_ERROR(expr) do{ \
    cudaError_t err = (expr); \
    if(err != cudaSuccess) \
    { printf("CUDA ERROR: %s %s at %s:%d\n", cudaGetErrorName(err), cudaGetErrorString(err), __FILE__, __LINE__); } \
    }while(false)

