.PHONY: pointer type macro clean

GCLANG = clang++

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

clean: 
	rm -rf pointer.o type.o macro.o