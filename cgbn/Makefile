.PHONY: pick clean download-gtest kepler maxwell pascal volta turing ampere check

gmpdir:=$(shell readlink -f ../../x86_64/gmp)

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
	make -C samples clean
	make -C unit_tests clean
	make -C perf_tests clean

download-gtest:
	wget 'https://github.com/google/googletest/archive/master.zip' -O googletest-master.zip
	unzip googletest-master.zip 'googletest-master/googletest/*'
	mv googletest-master/googletest gtest
	rmdir googletest-master
	rm -f googletest-master.zip

kepler: check
	make GMP_HOME=$(gmpdir) -C samples kepler
	make GMP_HOME=$(gmpdir) -C unit_tests kepler
	make GMP_HOME=$(gmpdir) -C perf_tests kepler

maxwell: check
	make GMP_HOME=$(gmpdir) -C samples maxwell
	# make GMP_HOME=$(gmpdir) -C unit_tests maxwell
	# make GMP_HOME=$(gmpdir) -C perf_tests maxwell

pascal: check
	make GMP_HOME=$(gmpdir) -C samples pascal
	make GMP_HOME=$(gmpdir) -C unit_tests pascal
	make GMP_HOME=$(gmpdir) -C perf_tests pascal

volta: check
	make GMP_HOME=$(gmpdir) -C samples volta
	make GMP_HOME=$(gmpdir) -C unit_tests volta
	make GMP_HOME=$(gmpdir) -C perf_tests volta

turing: check
	make GMP_HOME=$(gmpdir) -C samples turing
	make GMP_HOME=$(gmpdir) -C unit_tests turing
	make GMP_HOME=$(gmpdir) -C perf_tests turing

ampere: check
	make GMP_HOME=$(gmpdir) -C samples ampere
	make GMP_HOME=$(gmpdir) -C unit_tests ampere
	make GMP_HOME=$(gmpdir) -C perf_tests ampere

check:
	@if [ -z "$(GTEST_HOME)" -a ! -d "gtest" ]; then echo "Google Test framework required, see documentation"; exit 1; fi

# end of file
