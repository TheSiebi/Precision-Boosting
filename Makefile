.PHONY: prepareBuild build clean run

USER_FLAGS=
BASE_FLAGS=$(USER_FLAGS) -Wall -Wextra -Wpedantic -g -ffp-contract=off 
OPT_FLAGS=$(BASE_FLAGS) -O3
DEBUG_FLAGS=$(BASE_FLAGS) -O0 -fsanitize=address

CC=gcc

OBJ_FILES=build/timer.o build/cJSON.o build/matmul_v0.o 


.PHONY: $(OBJ_FILES)


build: prepareBuild $(OBJ_FILES)
	$(CC) $(OPT_FLAGS) src/main.c $(OBJ_FILES) -o build/main -lm

prepareBuild:
	mkdir -p build
	cp -n -T machine.template src/machine.h

build/timer.o: src/timer.c
	$(CC) $(OPT_FLAGS) -c src/timer.c -o $@

build/cJSON.o: lib/cjson/cJSON.c
	$(CC) $(OPT_FLAGS) -c lib/cjson/cJSON.c -o $@
	
build/matmul_v0.o:
	$(CC) $(OPT_FLAGS) -c src/impls/matmul_v0.c -o $@

run:
	./build/main

plot:
	bash ./plot.sh

clean:
	rm -r ./build
