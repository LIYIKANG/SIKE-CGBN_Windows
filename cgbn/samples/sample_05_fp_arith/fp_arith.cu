/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/cpu_support.h"
#include "../utility/cpu_simple_bn_math.h"
#include "../utility/gpu_support.h"

#include <cassert>
#include <iostream>

/************************************************************************************************
 *  This example performs component-wise addition of two arrays of 1024-bit bignums.
 *
 *  The example uses a number of utility functions and macros:
 *
 *    random_words(uint32_t *words, uint32_t num_instances)
 *       fills words[0 .. num_instances-1] with random data
 *
 *    add_words(uint32_t *r, uint32_t *a, uint32_t *b, uint32_t num_instances)
 *       sets bignums r = a+b, where r, a, and b are num_instances words in length
 *
 *    compare_words(uint32_t *a, uint32_t *b, uint32_t num_instances)
 *       compare bignums a and b, where a and b are num_instances words in length.
 *       return 1 if a>b, 0 if a==b, and -1 if b>a
 *
 *    CUDA_CHECK(call) is a macro that checks a CUDA result for an error,
 *    if an error is present, it prints out the error, call, file and line.
 *
 *    CGBN_CHECK(report) is a macro that checks if a CGBN error has occurred.
 *    if so, it prints out the error, and instance information
 *
 ************************************************************************************************/

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 32
#define NLIMBS ((434+31)/32)
#define BITS (32*NLIMBS)
#define NUM_INSTANCES 10

typedef enum {
  add_ope, sub_ope, mul_ope, div_ope, madd_ope,
  cx1_ope, // c = a*b + (-a)*(-b)
} operation_t;
#define OPE madd_ope

const char* p434 = "0x2341F271773446CFC5FD681C520567BC65C783158AEA3FDC1767AE2FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";

// Declare the instance type
typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> c;
  cgbn_mem_t<BITS> d;
  cgbn_mem_t<BITS> e;
  cgbn_mem_t<BITS> f;
  cgbn_mem_t<BITS> z;
  cgbn_mem_t<BITS> p;
} instance_t;


// support routine to generate random instances
instance_t*
generate_instances(const uint32_t num_instances) {
  const size_t bitlen = how_many_bits(p434);
  const uint32_t nlimbs = (bitlen+31) / 32;
  assert(nlimbs == NLIMBS);
  assert(nlimbs*32 == BITS);

  mpz_t p;
  mpz_init(p);
  mpz_set_str(p, p434, 0);
  instance_t* instances=(instance_t*)malloc(sizeof(instance_t)*num_instances);
  for (uint32_t i = 0; i < num_instances; ++i) {
    fp_random(instances[i].a._limbs, nlimbs, p);
    fp_random(instances[i].b._limbs, nlimbs, p);
    fp_random(instances[i].c._limbs, nlimbs, p);
    from_mpz(p, instances[i].p._limbs, nlimbs);
  }
  mpz_clear(p);
  fp_random(nullptr, nlimbs, p);
  return instances;
} // generate_instances


// support routine to verify the GPU results using the CPU
void
verify_results(instance_t* instances, const uint32_t num_instances) {
  uint32_t** correct = (uint32_t**)malloc(sizeof(uint32_t*) * num_instances);
  assert(correct != NULL);
  for (uint32_t i = 0; i < num_instances; ++i) {
    correct[i] = (uint32_t*)malloc(sizeof(uint32_t) * NLIMBS);
    assert(correct[i] != NULL);
  }

  mpz_t a, b, c, d, e, f, z, p;
  mpz_inits(a, b, c, d, e, f, z, p, NULL);
  mpz_set_str(p, p434, 0);
  for (uint32_t i = 0; i < num_instances; ++i) {
    to_mpz(a, instances[i].a._limbs, NLIMBS);
    to_mpz(b, instances[i].b._limbs, NLIMBS);
    to_mpz(c, instances[i].c._limbs, NLIMBS);
    if (OPE == add_ope) { // z = a+b
      mpz_add(z, a, b); mpz_mod(z, z, p);
    } else if (OPE == sub_ope) { // z = a-b
      mpz_sub(z, a, b); mpz_mod(z, z, p);
      if (mpz_cmp_ui(z, 0U) < 0) { mpz_add(z, z, p); }
    } else if (OPE == mul_ope) { // z = a*b
      mpz_mul(z, a, b); mpz_mod(z, z, p);
    } else if (OPE == div_ope) { // z = a/b
      mpz_t inv_b;
      mpz_init(inv_b);
      mpz_invert(inv_b, b, p);
      mpz_mul(z, a, inv_b); mpz_mod(z, z, p);
      mpz_clear(inv_b);
    } else if (OPE == madd_ope) { // z = a + b*c
      mpz_t u;
      mpz_init(u);
      mpz_mul(u, b, c); mpz_mod(u, u, p);
      mpz_add(z, a, u);
      if (mpz_cmp(z, p) > 0) { mpz_sub(z, z, p); }
      mpz_clear(u);
    } else if (OPE == cx1_ope) { // c = a*b + (-a)*(-b)
      mpz_mul(z, a, b);
      mpz_mul_ui(z, z, 2);
      mpz_mod(z, z, p);
    } else {
      assert(false);
    }
    from_mpz(z, correct[i], NLIMBS);
  }
  for (uint32_t i = 0; i < num_instances; ++i) {
    for (int32_t j = 0; j < NLIMBS; ++j) {
      if (correct[i][j] != instances[i].z._limbs[j]) {
        std::cout << "NG: [" << i << "]["  << j << "]" << std::endl;
        std::cout << "correc " << correct[i][j] << std::endl;
        std::cout << "result " << instances[i].z._limbs[j] << std::endl;
        goto clean_up;
      }
    }
  }
  std::cout << "OK: all results" << std::endl;

clean_up:
  for (uint32_t i = 0; i < num_instances; ++i) { free(correct[i]); }
  free(correct);
  mpz_clears(a, b, c, d, e, f, z, p, NULL);
} // verify_results


// helpful typedefs for the kernel
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

// the actual kernel
__global__ void
kernel_add(cgbn_error_report_t *report, instance_t *instances, uint32_t num_instances) {
  // decode an instance number from the blockIdx and threadIdx
  int32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) { return; }

  context_t bn_context(cgbn_report_monitor, report, num_instances); // construct a context
  env_t bn_env(bn_context.env<env_t>()); // construct an environment
  typename cgbn_env_t<context_t, BITS>::cgbn_t a, b, c, z, p;

  cgbn_load(bn_env, p, &(instances[instance_idx].p));
  cgfp_load(bn_env, a, &(instances[instance_idx].a), p);
  cgfp_load(bn_env, b, &(instances[instance_idx].b), p);
  cgfp_load(bn_env, c, &(instances[instance_idx].c), p);
  if (OPE == add_ope) {
    cgfp_add(bn_env, z, a, b, p);
  } else if (OPE == sub_ope) {
    cgfp_sub(bn_env, z, a, b, p);
  } else if (OPE == mul_ope) {
    cgfp_mul(bn_env, z, a, b, p);
  } else if (OPE == div_ope) {
    cgfp_div(bn_env, z, a, b, p);
  } else if (OPE == madd_ope) {
    cgfp_madd(bn_env, z, a, b, c, p); 
  } else if (OPE == cx1_ope) {      
    typename cgbn_env_t<context_t, BITS>::cgbn_t neg_a, neg_b, u, v;
    cgfp_mul(bn_env, u, a, b, p);
    cgfp_negate(bn_env, neg_a, a, p);
    cgfp_negate(bn_env, neg_b, b, p);
    cgfp_mul(bn_env, v, neg_a, neg_b, p);
    cgfp_add(bn_env, c, u, v, p);
  } else {
    assert(false);
  }
  cgfp_store(bn_env, &(instances[instance_idx].z), z, p);
} // kernel_add


int main() {
  instance_t* instances = nullptr;
  instance_t* gpu_instances = nullptr;
  cgbn_error_report_t* report = nullptr;

  std::cout << "Genereating instances ..." << std::endl;
  instances = generate_instances(NUM_INSTANCES);

  std::cout << "Copying instances to the GPU ..." << std::endl;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpu_instances, sizeof(instance_t)*NUM_INSTANCES));
  CUDA_CHECK(cudaMemcpy(gpu_instances, instances, sizeof(instance_t)*NUM_INSTANCES, cudaMemcpyHostToDevice));

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  std::cout << "Running GPU kernel ..." << std::endl;
  // launch with 32 threads per instance, 128 threads (4 instances) per block
  kernel_add<<<(NUM_INSTANCES+3)/4, 128>>>(report, gpu_instances, NUM_INSTANCES);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  std::cout << "Copying results back to CPU ..." << std::endl;
  CUDA_CHECK(cudaMemcpy(instances, gpu_instances, sizeof(instance_t)*NUM_INSTANCES, cudaMemcpyDeviceToHost));

  std::cout << "Verifying the results ..." << std::endl;
  verify_results(instances, NUM_INSTANCES);

  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpu_instances));
  CUDA_CHECK(cgbn_error_report_free(report));

  return 0;
} // main

// end of file
