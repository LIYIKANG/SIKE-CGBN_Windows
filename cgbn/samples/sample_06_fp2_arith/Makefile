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


kepler:
	$(MAKE) --no-print-directory GPUARCH=sm_35 build
maxwell:
	$(MAKE) --no-print-directory GPUARCH=sm_50 build
pascal:
	$(MAKE) --no-print-directory GPUARCH=sm_60 build
volta:
	$(MAKE) --no-print-directory GPUARCH=sm_70 build
turing:
	$(MAKE) --no-print-directory GPUARCH=sm_75 build
ampere:
	$(MAKE) --no-print-directory GPUARCH=sm_80 build

build:
	$(MAKE) --no-print-directory add
	$(MAKE) --no-print-directory sub
	$(MAKE) --no-print-directory mul
	$(MAKE) --no-print-directory div
	$(MAKE) --no-print-directory madd

add:
	$(MAKE) --no-print-directory OPE=0 OPESTR=add fp2_arith-add
sub:
	$(MAKE) --no-print-directory OPE=1 OPESTR=sub fp2_arith-sub
mul:
	$(MAKE) --no-print-directory OPE=2 OPESTR=mul fp2_arith-mul
div:
	$(MAKE) --no-print-directory OPE=3 OPESTR=div fp2_arith-div
madd:
	$(MAKE) --no-print-directory OPE=4 OPESTR=madd fp2_arith-madd
	
fp2_arith-$(OPESTR): fp2_arith.cu
	nvcc -DOPE=$(OPE) --verbose --std c++14 -Wno-deprecated-gpu-targets $(INC) $(LIB) -I../../include -arch=$(GPUARCH) $^ -o $@ -lgmp

check: fp2_arith-add fp2_arith-sub fp2_arith-mul fp2_arith-div fp2_arith-madd
	./fp2_arith-add 
	./fp2_arith-sub 
	./fp2_arith-mul 
	./fp2_arith-div 
	./fp2_arith-madd

clean:
	rm -f ./fp2_arith-add ./fp2_arith-sub ./fp2_arith-mul ./fp2_arith-div ./fp2_arith-madd

# end of file
