.PHONY: pointer type macro cuda clean

GCLANG = clang++
GCC = g++
NVCC = nvcc
CUDA = vector_add

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

cuda:
	@$(NVCC) $(CUDA).cu -o $(CUDA).o
	@./$(CUDA).o
	@rm -rf $(CUDA).o


clean: 
	rm -rf pointer.o type.o macro.o

