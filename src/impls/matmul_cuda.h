
template<typename InputType, typename OutputType, int version>
void matmulTensorCores(InputType *A, InputType *B, OutputType *C, int M, int K, int N);

template<typename InputType, typename OutputType, int version>
void matmulCUDACores(InputType *A, InputType *B, OutputType *C, int M, int K, int N);

