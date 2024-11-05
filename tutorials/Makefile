.PHONY: pointer type macro add clean

GCLANG = clang++
GCC = g++
NVCC = nvcc

type:
	@$(GCLANG) type.cpp -o type.o
	@./type.o
	@rm -rf type.o

pointer:
	@$(GCLANG) pointer.cpp -o pointer.o
	@./pointer.o
	@rm -rf pointer.o

macro:
	@$(GCLANG) macro.cpp -o macro.o
	@./macro.o
	@rm -rf macro.o

add:
	@$(NVCC) vector_add.cu -o vector_add.o
	@./vector_add.o
	@rm -rf vector_add.o

clean: 
	rm -rf pointer.o type.o macro.o

