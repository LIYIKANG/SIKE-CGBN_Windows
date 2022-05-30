//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//
#include <memory.h>
#include <stdint.h>
#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>

#include "helper_cuda.h"

#include "sike_params.cuh"
#include "test_arith.cuh"

#pragma comment(lib, "libgmp.dll.a")

#define P434

static void 
help(const std::string& program_name) {
  std::cout << "Usage: " << program_name;
  std::cout << " {arith|sidh_int|pke|sike|sike_int|sike_speed|sike_speed_int}";
  std::cout << " {num}";
  std::cout << std::endl;
}


int main(int argc, char *argv[]) {  

findCudaDevice(argc, (const char **)argv);
  
#if defined(P434)
  const sike_params_raw_t params_raw = SIKEp434;
#elif defined(P503)
  const sike_params_raw_t params_raw = SIKEp503;
#elif defined(P610)    
  const sike_params_raw_t params_raw = SIKEp610;
#elif defined(P751)  
  const sike_params_raw_t params_raw = SIKEp751;
#else
  #error Define P434, P503, P610, or P751
#endif
 
  // true: all tests are passed, false: some test is failed.
  bool rc = test_arith(params_raw);
  return (rc == true) ? 0 : 1;
}
