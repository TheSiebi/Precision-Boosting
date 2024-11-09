.PHONY: prepareBuild build clean run

USER_FLAGS=
BASE_FLAGS=$(USER_FLAGS) -Wall -Wextra -Wpedantic -g -ffp-contract=off
CPP_FLAGS=$(BASE_FLAGS) -std=gnu++2a -Wno-missing-field-initializers
OPT_FLAGS=$(CPP_FLAGS) -O3
DEBUG_FLAGS=$(CPP_FLAGS) -O0 -fsanitize=address
CUDA_FLAGS= -g -arch=sm_70
CUDA_DEBUG_FLAGS=$(CUDA_FLAGS) -G -Xptxas -v

CC=gcc
CPP=g++

OBJ_FILES=build/profiler.o build/timer.o build/rand.o build/cJSON.o 
OBJ_FILES+=build/matmul_cuda_v0.o build/split_v0.o build/merge_accumulate_v0.o
OBJ_FILES+=build/matmul_simpleMarkidis_v0.o
OBJ_FILES+=build/matmul_simpleOotomo_v0.o
OBJ_FILES+=build/matmul_Ootomo.o
OBJ_FILES+=build/matmul_cuBLAS.o


.PHONY: $(OBJ_FILES)


build: prepareBuild $(OBJ_FILES)
	$(CPP) $(OPT_FLAGS) src/main.cpp $(OBJ_FILES) -o build/main -lm -lcudart -lcublas

prepareBuild:
	mkdir -p build
	cp -n -T machine.template src/machine.h

build/profiler.o: src/profiler.c
	$(CC) $(BASE_FLAGS) -O3 -c src/profiler.c -o $@

build/timer.o: src/timer.cpp
	$(CPP) $(OPT_FLAGS) -c src/timer.cpp -o $@

build/rand.o: src/rand.cpp
	$(CPP) $(OPT_FLAGS) -c src/rand.cpp -o $@

build/cJSON.o: lib/cjson/cJSON.c
	$(CC) $(BASE_FLAGS) -O3 -c lib/cjson/cJSON.c -o $@
	
build/matmul_cuda_v0.o:
	nvcc $(CUDA_FLAGS) -c src/impls/matmul_cuda_v0.cu -o $@

build/matmul_simpleMarkidis_v0.o:
	nvcc $(CUDA_DEBUG_FLAGS) -c src/impls/matmul_simpleMarkidis_v0.cu -o $@

build/matmul_simpleOotomo_v0.o:
	nvcc $(CUDA_FLAGS) -c src/impls/matmul_simpleOotomo_v0.cu -o $@

build/matmul_Ootomo.o:
	nvcc $(CUDA_FLAGS) -c src/impls/matmul_Ootomo.cu -o $@

build/matmul_cuda_v1.o:
	nvcc $(CUDA_FLAGS) -c src/impls/matmul_cuda_v1.cu -o $@

build/split_v0.o:
	nvcc $(CUDA_FLAGS) -c src/impls/split_v0.cu -o $@

build/merge_accumulate_v0.o:
	nvcc $(CUDA_FLAGS) -c src/impls/merge_accumulate_v0.cu -o $@

build/matmul_cuBLAS.o:
	nvcc $(CUDA_FLAGS) -c src/impls/matmul_cuBLAS.cu -o $@

run:
	./build/main

plot:
	bash ./plot.sh

clean:
	rm -r ./build
