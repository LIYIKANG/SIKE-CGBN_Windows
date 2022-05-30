gmpdir := $(shell readlink -f ../../../../x86_64/gmp)
GMP_HOME := $(gmpdir)

ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

pick:
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo "   make turing"
	@echo "   make ampere"
	@echo

clean:
	rm -f fp2_time

kepler:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_35 fp2_time.cu -o fp2_time -lgmp

maxwell:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_50 fp2_time.cu -o fp2_time -lgmp

pascal:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_60 fp2_time.cu -o fp2_time -lgmp

volta: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_70 fp2_time.cu -o fp2_time -lgmp

turing: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_75 fp2_time.cu -o fp2_time -lgmp

ampere: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_80 fp2_time.cu -o fp2_time -lgmp
