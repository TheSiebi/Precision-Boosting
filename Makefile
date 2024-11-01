.PHONY: prepareBuild build clean run

USER_FLAGS=
BASE_FLAGS=$(USER_FLAGS) -Wall -Wextra -Wpedantic -g -ffp-contract=off
CPP_FLAGS=$(BASE_FLAGS) -std=c++20 -Wno-missing-field-initializers
OPT_FLAGS=$(CPP_FLAGS) -O3
DEBUG_FLAGS=$(CPP_FLAGS) -O0 -fsanitize=address
CUDA_FLAGS= -g

CC=gcc

OBJ_FILES=build/profiler.o build/timer.o build/cJSON.o 
OBJ_FILES+=build/matmul_cuda_v0.o build/split_v0.o build/merge_accumulate_v0.o
OBJ_FILES+=build/matmul_simpleMarkidis_v0.o


.PHONY: $(OBJ_FILES)


build: prepareBuild $(OBJ_FILES)
	$(CC) $(OPT_FLAGS) src/main.cpp $(OBJ_FILES) -o build/main -lm -lcudart

prepareBuild:
	mkdir -p build
	cp -n -T machine.template src/machine.h

build/profiler.o: src/profiler.c
	$(CC) $(BASE_FLAGS) -O3 -c src/profiler.c -o $@

build/timer.o: src/timer.cpp
	$(CC) $(OPT_FLAGS) -c src/timer.cpp -o $@

build/cJSON.o: lib/cjson/cJSON.c
	$(CC) $(BASE_FLAGS) -O3 -c lib/cjson/cJSON.c -o $@
	
build/matmul_cuda_v0.o:
	nvcc $(CUDA_FLAGS) -c src/impls/matmul_cuda_v0.cu -o $@

build/matmul_simpleMarkidis_v0.o:
	nvcc $(CUDA_FLAGS) -c src/impls/matmul_simpleMarkidis_v0.cu -o $@

build/matmul_cuda_v1.o:
	nvcc $(CUDA_FLAGS) -c src/impls/matmul_cuda_v1.cu -o $@

build/split_v0.o:
	nvcc $(CUDA_FLAGS) -c src/impls/split_v0.cu -o $@

build/merge_accumulate_v0.o:
	nvcc $(CUDA_FLAGS) -c src/impls/merge_accumulate_v0.cu -o $@

run:
	./build/main

plot:
	bash ./plot.sh

clean:
	rm -r ./build
