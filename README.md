# Emulating Double-Precision Floating-Point Matrix Multiplication Using Half-Precision Floating-Point Numbers

**Marc Matter, Max Striebel, Michael Mörschell, Michael Siebenmann, Tynan Richards**

> This is a course project for ETH's Design of Parallel and High-Performance Computing (DPHPC) course. We implement fp64 matrix multiplication using fp16 and fp32 tensor cores.

## Usage
The build system is currently configured to operate on Unix machines.

To **build** use the command: `make build`

To **run** use the command: `make run` (by default this will validate the correctness of all implementation versions specified in the main function of main.c)

To **generate** performance and speedup **plots** use the command: `make plot`


## CUDA Installation (WSL2)

1. Follow steps 2 and 3 of the [Official Installation Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). 
2. Add the following lines at the end of ~/.profile (check that the paths match on your system):
   ``` 
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
   ```