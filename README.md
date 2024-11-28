# Emulating Double-Precision Floating-Point Matrix Multiplication Using Half-Precision Floating-Point Numbers

**Marc Matter, Max Striebel, Michael Mörschell, Michael Siebenmann, Tynan Richards**

> This is a course project for ETH's Design of Parallel and High-Performance Computing (DPHPC) course. We implement fp64 matrix multiplication using fp16 and fp32 tensor cores.

## Usage
The build system is currently configured to operate on Unix machines.

To **build** use the command: `make build`

To **run** use the command: `make run` (by default this will validate the correctness of all implementation versions specified in the main function of main.c)

To **generate** performance and speedup **plots** use the command: `make plot`
Plot generation uses Python 3.10.12 and requires matplotlib, numpy and scipy.

## CUDA Installation (WSL2)

1. Follow steps 2 and 3 of the [Official Installation Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). 
2. Add the following lines at the end of ~/.profile (check that the paths match on your system):
   ``` 
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
   ```
3. For debugging, execute the following command on Windows
   ```
   REG ADD “HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger” /f /v EnableInterface /t REG_DWORD /d 1
   ```
