SUBDIRS := sample_01_add sample_02_modinv sample_03_powm sample_04_miller_rabin sample_05_fp_arith sample_06_fp2_arith sample_07_fp2_time

TARGETS := clean kepler maxwell pascal volta turing ampere
gmpdir := $(shell readlink -f ../../../x86_64/gmp)

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

$(TARGETS): $(SUBDIRS)
$(SUBDIRS):
	make GMP_HOME=$(gmpdir) -C $@ $(MAKECMDGOALS)

.PHONY: $(TARGETS) $(SUBDIRS)

# end file
