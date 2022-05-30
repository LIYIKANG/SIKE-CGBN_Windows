//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//
// Based on https://github.com/Microsoft/PQCrypto-SIDH
//


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include <string>

#include <cuda.h>
#include <gmp.h>

#include "cgbn/cgbn.h"
#include "cgbn_ext.cuh"
#include "gpu_support.h"
#include "assume.h"
#include "sike_params.cuh"
#include "fp.cuh"
#include "test_arith.cuh"

//#pragma comment(lib, "libgmp.dll.a")

#define P434

// IMPORTANT: DO NOT DEFINE BITS OR TPI BEFORE INCLUDING CGBN
// Threads per instance
const uint32_t TPI = 32;
// A Minimum Multiple of 32 that is not Less than x, 32 is the number of bits of a word (int).
#define MML(x) ((((x)+(32-1))/32)*32)
#if defined(P434)
  const uint32_t BITS = MML(434);
#elif defined(P503)
  const uint32_t BITS = MML(503)
#elif defined(P610)
  const uint32_t BITS = MML(503)
#elif defined(P751)
  const uint32_t BITS = MML(751)
#else
  #error Not defined: P434, P503, P610, and P751
#endif // defined(P434), defined(P503), defined(P610), or defined(P751)


// ================================================
#if 1 // test for GF(p)-arithmetic operations
// Declare the instance type
struct fp_instance_t {
  cgbn_mem_t<BITS> p; // modulus
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> c;
  cgbn_mem_t<BITS> d;
  cgbn_mem_t<BITS> e;
  cgbn_mem_t<BITS> f;
  bool is_equal;
};


__host__ static void
fp_generate_instances(fp_instance_t* host_instances,
    const uint32_t num_instances,
    std::string p) {
  mpz_t p_mpz, r_mpz;
  mpz_inits(p_mpz, r_mpz, NULL);
  // 0 means that leading characters are used for the base.
  mpz_set_str(p_mpz, p.c_str(), 0);
  //ASSUME(mpz_sizeinbase(p_mpz, 2) <= BITS);

  gmp_randstate_t state;
  gmp_randinit_default(state);
  const uint32_t nlimbs = (BITS+31) / 32;
  for(uint32_t i = 0; i < num_instances; ++i) {
    convert_mpz_to_array(p_mpz, nlimbs, host_instances[i].p._limbs);
    mpz_urandomm(r_mpz, state, p_mpz);
    convert_mpz_to_array(r_mpz, nlimbs, host_instances[i].a._limbs);
    mpz_urandomm(r_mpz, state, p_mpz);
    convert_mpz_to_array(r_mpz, nlimbs, host_instances[i].b._limbs);
    mpz_urandomm(r_mpz, state, p_mpz);
    convert_mpz_to_array(r_mpz, nlimbs, host_instances[i].c._limbs);
    mpz_urandomm(r_mpz, state, p_mpz);
    convert_mpz_to_array(r_mpz, nlimbs, host_instances[i].d._limbs);
    mpz_urandomm(r_mpz, state, p_mpz);
    convert_mpz_to_array(r_mpz, nlimbs, host_instances[i].e._limbs);
    mpz_urandomm(r_mpz, state, p_mpz);
    convert_mpz_to_array(r_mpz, nlimbs, host_instances[i].f._limbs);
    host_instances[i].is_equal = false;
  }
  gmp_randclear(state);
  mpz_clears(p_mpz, r_mpz, NULL);
} // fp_generate_instances


static void
fp_free_instances(fp_instance_t* instances,
    const uint32_t num_instances) {
  // instances[i].a._limbs is an array, that is, is not memory allocated with malloc.
  free(instances);
}


// support routine to verify the GPU results using the CPU
// true: ok, false: ng
static bool
fp_is_equal(const fp_instance_t* host_instances,
    const uint32_t num_instances) {
  for (uint32_t i = 0; i < num_instances; ++i) {
    if (host_instances[i].is_equal == false) {
      return false;
    }
  }
  return true;
}


// 1: e = (a+b)+c, // f = a+(b+c)
__global__ static void
check_add1(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }

  cgbn_error_report_t *report;
   // create a cgbn_error_report for CGBN to report back errors
  //CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  typedef cgbn_context_t<TPI, cgbn_default_parameters_t> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, instance_idx);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Add(bn_env, p, a, b, d);
  fp_Add(bn_env, p, d, c, e);
  fp_Add(bn_env, p, b, c, d);
  fp_Add(bn_env, p, a, d, f);
  fp_IsEqual(bn_env, p, e, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_add1


// 2: e = a+b, f = b+a
__global__ void
check_add2(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Add(bn_env, p, a, b, e);
  fp_Add(bn_env, p, b, a, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_add2


// 3: b = 0, e = a+b = a+0 = a, // f = a
__global__ void
check_add3(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Zero(bn_env, p, b);
  fp_Add(bn_env, p, a, b, e);
  fp_Copy(bn_env, p, f, a);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_add3


// 4: e = 0, d = -a, f = a+d = a+(-a) = 0
__global__ void
check_add4(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Zero(bn_env, p, e);
  fp_Negative(bn_env, p, a, d);
  fp_Add(bn_env, p, a, d, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_add4


// sub1: e = (a-b)-c, f = a-(b+c)
__global__ void
check_sub1(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Subtract(bn_env, p, a, b, d);
  fp_Subtract(bn_env, p, d, c, e);
  fp_Add(bn_env, p, b, c, d);
  fp_Subtract(bn_env, p, a, d, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_sub1


// sub2: f = a-b, e = -(b-a)
__global__ void
check_sub2(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Subtract(bn_env, p, a, b, f);
  fp_Subtract(bn_env, p, b, a, e);
  fp_Negative(bn_env, p, e, e);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_sub2


// sub3: e = a, d = 0, f = a-d = a
__global__ void
check_sub3(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Copy(bn_env, p, e, a);
  fp_Zero(bn_env, p, d);
  fp_Subtract(bn_env, p, a, d, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_sub3


// mul1: e = (a*b)*c, f = a*(b*c)
__global__ void
check_mul1(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Multiply(bn_env, p, a, b, d);
  fp_Multiply(bn_env, p, d, c, e);
  fp_Multiply(bn_env, p, b, c, d);
  fp_Multiply(bn_env, p, a, d, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_mul1


// mul2: e = a*(b+c), f = a*b+a*c
__global__ void
check_mul2(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Add(bn_env, p, b, c, d);
  fp_Multiply(bn_env, p, a, d, e);
  fp_Multiply(bn_env, p, a, b, d);
  fp_Multiply(bn_env, p, a, c, f);
  fp_Add(bn_env, p, d, f, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_mul2


// mul3: e = a*b, f = b*a
__global__ void
check_mul3(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Multiply(bn_env, p, a, b, e);
  fp_Multiply(bn_env, p, b, a, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_mul3


// mul4: b = 1, e = a*b, f = a
__global__ void
check_mul4(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Constant(bn_env, p, 1U, b);
  fp_Multiply(bn_env, p, a, b, e);
  fp_Copy(bn_env, p, f, a);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_mul4


// mul5: e = 0, f = a*b
__global__ void
check_mul5(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Constant(bn_env, p, 0U, e);
  fp_Multiply(bn_env, p, a, e, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_mul5


// sqr1: e = a^2, f = a*a
__global__ void
check_sqr1(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Square(bn_env, p, a, e);
  fp_Multiply(bn_env, p, a, a, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_sqr1


// sqr2: e = 0, f = e^2 = 0
__global__ void
check_sqr2(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Zero(bn_env, p, e);
  fp_Square(bn_env, p, e, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_sqr2


// inv1: e = 1, f = a^{-1}, f = a*f
__global__ void
check_inv1(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Constant(bn_env, p, 1U, e);
  fp_Copy(bn_env, p, f, a);
  fp_Invert(bn_env, p, f, f);
  fp_Multiply(bn_env, p, a, f, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_inv1


// inv2: e = a, f = a^{-1}, f = f^{-1} = a^{-1}^{-1} = a
__global__ void
check_inv2(fp_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p, a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgbn_load(bn_env, a, &(device_instances[instance_idx].a));
  cgbn_load(bn_env, b, &(device_instances[instance_idx].b));
  cgbn_load(bn_env, c, &(device_instances[instance_idx].c));
  cgbn_load(bn_env, d, &(device_instances[instance_idx].d));
  cgbn_load(bn_env, e, &(device_instances[instance_idx].e));
  cgbn_load(bn_env, f, &(device_instances[instance_idx].f));
  fp_Copy(bn_env, p, e, a);
  fp_Invert(bn_env, p, a, f);
  fp_Invert(bn_env, p, f, f);
  device_instances[instance_idx].is_equal = fp_IsEqual(bn_env, p, e, f);
} // check_inv1


__host__ static bool
test_fp(const sike_params_raw_t& params_raw,
    const uint32_t num_instances) {
  //CUDA_CHECK(cudaSetDevice(0));
  // launch with 32 threads per instance, 128 threads (4 instances) per block
  const uint32_t size_of_instances = sizeof(fp_instance_t) * num_instances;
  const uint32_t ngrid = (num_instances + 3) / 4;
  bool rc = false;
  // add1: e = (a+b)+c, f = a+(b+c)
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
   // ASSUME(host_instances != nullptr);

    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_add1<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // add2: e = a+b, f = b+a
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_add2<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // add3: e=a+0, f=a
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_add3<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // add4: e = 0, d = -a, f = a+d = a+(-a) = 0
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_add4<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // sub1: e = (a-b)-c, f = a-(b+c)
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_sub1<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // sub2: d = a-b, e = -(b-a)
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_sub2<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // sub3: f = 0, e = a+(-a)
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_sub3<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // mul1: e = (a*b)*c, f = a*(b*c)
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_mul1<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // mul2: e = a*(b+c), f = a*b+a*c
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_mul2<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // mul3: e = a*b, f = b*a
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_mul3<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // mul4: b = 1, e = a*b, f = a
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_mul4<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // mul5: e = 0, f = a*b
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_mul5<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // sqr1: e = a^2, f = a*a
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_sqr1<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // sqr2: e = 0, f = e^2 = 0
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_sqr2<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // inv1: e = 1, f = a^{-1}, f = a*f
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_inv1<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // inv2: e = a, f = a^{-1}, f = f^{-1}
  if (true) {
    fp_instance_t* host_instances = new fp_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_inv2<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp_is_equal(host_instances, num_instances);
    fp_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  return true;
} // test_fp
#elif 0 // for debug
__host__ static bool
test_fp(const sike_params_raw_t& params_raw,
    const uint32_t num_instances) {
  return true;
}
#else
  #error "Not defined: test_fp()"
#endif // test for GF(p)-arithmetic operations


// ================================================
// Check GF(p^2)-arithmetic operations
#if 1 // test for GF(p^2)-arithmetic operations
// Declare the instance type
struct fp2_instance_t {
  cgbn_mem_t<BITS> p; // modulus
  cgbn2_mem_t<BITS> a;
  cgbn2_mem_t<BITS> b;
  cgbn2_mem_t<BITS> c;
  cgbn2_mem_t<BITS> d;
  cgbn2_mem_t<BITS> e;
  cgbn2_mem_t<BITS> f;
  bool is_equal;
};


__host__ static void
fp2_generate_instances(fp2_instance_t* host_instances,
    const uint32_t num_instances,
    const std::string& p) {
  const bool debug = true;
  mpz_t p_mpz, r0_mpz, r1_mpz;
  mpz_inits(p_mpz, r0_mpz, r1_mpz, NULL);
  // 0 means that leading characters are used for the base.
  mpz_set_str(p_mpz, p.c_str(), 0);
  //ASSUME(mpz_sizeinbase(p_mpz, 2) <= BITS);
  gmp_randstate_t state;
  gmp_randinit_default(state);
  const uint32_t nlimbs = (BITS+31) / 32;
  for (uint32_t i = 0; i < num_instances; ++i) {
    convert_mpz_to_array(p_mpz, nlimbs, host_instances[i].p._limbs);

    mpz_urandomm(r0_mpz, state, p_mpz);
    mpz_urandomm(r1_mpz, state, p_mpz);
    if (!debug) {
      mpz_set_ui(r0_mpz, rand()&0xF);
      mpz_set_ui(r1_mpz, rand()&0xF);
    }
    convert_mpz_to_array(r0_mpz, nlimbs, host_instances[i].a.x0._limbs);
    convert_mpz_to_array(r1_mpz, nlimbs, host_instances[i].a.x1._limbs);

    mpz_urandomm(r0_mpz, state, p_mpz);
    mpz_urandomm(r1_mpz, state, p_mpz);
    if (!debug) {
      mpz_set_ui(r0_mpz, rand()&0xF);
      mpz_set_ui(r1_mpz, rand()&0xF);
    }
    convert_mpz_to_array(r0_mpz, nlimbs, host_instances[i].b.x0._limbs);
    convert_mpz_to_array(r1_mpz, nlimbs, host_instances[i].b.x1._limbs);

    mpz_urandomm(r0_mpz, state, p_mpz);
    mpz_urandomm(r1_mpz, state, p_mpz);
    if (!debug) {
      mpz_set_ui(r0_mpz, rand()&0xF);
      mpz_set_ui(r1_mpz, rand()&0xF);
    }
    convert_mpz_to_array(r0_mpz, nlimbs, host_instances[i].c.x0._limbs);
    convert_mpz_to_array(r1_mpz, nlimbs, host_instances[i].c.x1._limbs);

    mpz_urandomm(r0_mpz, state, p_mpz);
    mpz_urandomm(r1_mpz, state, p_mpz);
    if (!debug) {
      mpz_set_ui(r0_mpz, rand()&0xF);
      mpz_set_ui(r1_mpz, rand()&0xF);
    }
    convert_mpz_to_array(r0_mpz, nlimbs, host_instances[i].d.x0._limbs);
    convert_mpz_to_array(r1_mpz, nlimbs, host_instances[i].d.x1._limbs);

    mpz_urandomm(r0_mpz, state, p_mpz);
    mpz_urandomm(r1_mpz, state, p_mpz);
    if (!debug) {
      mpz_set_ui(r0_mpz, rand()&0xF);
      mpz_set_ui(r1_mpz, rand()&0xF);
    }
    convert_mpz_to_array(r0_mpz, nlimbs, host_instances[i].e.x0._limbs);
    convert_mpz_to_array(r1_mpz, nlimbs, host_instances[i].e.x1._limbs);

    mpz_urandomm(r0_mpz, state, p_mpz);
    mpz_urandomm(r1_mpz, state, p_mpz);
    if (debug) {
      mpz_set_ui(r0_mpz, rand()&0xF);
      mpz_set_ui(r1_mpz, rand()&0xF);
    }
    convert_mpz_to_array(r0_mpz, nlimbs, host_instances[i].f.x0._limbs);
    convert_mpz_to_array(r1_mpz, nlimbs, host_instances[i].f.x1._limbs);

    host_instances[i].is_equal = false;
  }
  gmp_randclear(state);
  mpz_clears(p_mpz, r0_mpz, r1_mpz, NULL);
} // fp2_generate_instances


__host__ static void
fp2_free_instances(fp2_instance_t* instances,
    const uint32_t num_instances) {
  // instances[i].a.x0_limbs is an array, that is, is not memory allocated with malloc.
  free(instances);
} // fp2_free_instances


// print an array as mpz_t with a prefix string.
static void
println_value_(const std::string& str,
    const uint32_t* limbs,
    const uint32_t nlimbs) {
  std::cout << str;
  mpz_t v;
  mpz_init(v);
  convert_array_to_mpz(nlimbs, limbs, v);
  mpz_out_str(stdout, 10, v);
  mpz_clear(v);
  std::cout << std::endl;
}


// support routine to verify the GPU results using the CPU
// true: ok, false: ng
__host__ static bool
fp2_is_equal(const fp2_instance_t* host_instances,
    const uint32_t num_instances) {
  const bool debug = true;
  for (uint32_t i = 0; i < num_instances; ++i) {
    if (!debug) {
      const uint32_t nlimbs = BITS / 32;
      if (true) std::cout << "i    " << i << std::endl;
      if (true) println_value_("p    ", host_instances[i].p._limbs, nlimbs);
      if (true) println_value_("a.x0 ", host_instances[i].a.x0._limbs, nlimbs);
      if (true) println_value_("a.x1 ", host_instances[i].a.x1._limbs, nlimbs);
      if (true) println_value_("b.x0 ", host_instances[i].b.x0._limbs, nlimbs);
      if (true) println_value_("b.x1 ", host_instances[i].b.x1._limbs, nlimbs);
      if (true) println_value_("c.x0 ", host_instances[i].c.x0._limbs, nlimbs);
      if (true) println_value_("c.x1 ", host_instances[i].c.x1._limbs, nlimbs);
      if (true) println_value_("d.x0 ", host_instances[i].d.x0._limbs, nlimbs);
      if (true) println_value_("d.x1 ", host_instances[i].d.x1._limbs, nlimbs);
      if (true) println_value_("e.x0 ", host_instances[i].e.x0._limbs, nlimbs);
      if (true) println_value_("e.x1 ", host_instances[i].e.x1._limbs, nlimbs);
      if (true) println_value_("f.x0 ", host_instances[i].f.x0._limbs, nlimbs);
      if (true) println_value_("f.x1 ", host_instances[i].f.x1._limbs, nlimbs);
    }
    if (host_instances[i].is_equal == false) {
      return false;
    }
  }
  return true;
}


// 1: e = (a+b)+c, // f = a+(b+c)
__global__ static void
check_fp2_add1(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
   cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Add(bn_env, p, a, b, d);
  fp2_Add(bn_env, p, d, c, e);
  fp2_Add(bn_env, p, b, c, d);
  fp2_Add(bn_env, p, a, d, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_add1


// add2: e = a+b, f = b+a
__global__ static void
check_fp2_add2(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Add(bn_env, p, a, b, e);
  fp2_Add(bn_env, p, b, a, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_add2


// add3: e = a, f = 0, f = a+f
__global__ static void
check_fp2_add3(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Copy(bn_env, p, e, a);
  fp2_Set(bn_env, p, f, 0U, 0U);
  fp2_Add(bn_env, p, a, f, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_add3


// add4: e = 0, f = -a, f = f+a
__global__ static void
check_fp2_add4(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Set(bn_env, p, e, 0U, 0U);
  fp2_Negative(bn_env, p, a, f);
  fp2_Add(bn_env, p, f, a, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_add4


// sub1: e = (a-b)-c, f = a-(b+c)
__global__ static void
check_fp2_sub1(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Sub(bn_env, p, a, b, e);
  fp2_Sub(bn_env, p, e, c, e);
  fp2_Add(bn_env, p, b, c, f);
  fp2_Sub(bn_env, p, a, f, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_sub1


// sub2: e = a-b, f = b-a, f = -f
__global__ static void
check_fp2_sub2(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Sub(bn_env, p, a, b, e);
  fp2_Sub(bn_env, p, b, a, f);
  fp2_Negative(bn_env, p, f, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_sub2


// sub3: e = a, f = 0, f = a-f
__global__ static void
check_fp2_sub3(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Copy(bn_env, p, a, e);
  fp2_Set(bn_env, p, f, 0U, 0U);
  fp2_Sub(bn_env, p, a, f, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_sub3


// sub4: e = 0, f = -a, f = a+f
__global__ static void
check_fp2_sub4(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Set(bn_env, p, e, 0U, 0U);
  fp2_Negative(bn_env, p, a, f);
  fp2_Add(bn_env, p, a, f, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_sub4


// mul1: e = (a*b)*c, f = a*(b*c)
__global__ static void
check_fp2_mul1(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Multiply(bn_env, p, a, b, e);
  fp2_Multiply(bn_env, p, e, c, e);
  fp2_Multiply(bn_env, p, b, c, f);
  fp2_Multiply(bn_env, p, a, f, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_mul1


// mul2: e = a*(b+c), f = a*b+a*c
__global__ static void
check_fp2_mul2(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Add(bn_env, p, b, c, e);
  fp2_Multiply(bn_env, p, a, e, e);
  fp2_Multiply(bn_env, p, a, b, d);
  fp2_Multiply(bn_env, p, a, c, f);
  fp2_Add(bn_env, p, d, f, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_mul2


// mul3: e = a*b, f = b*a
__global__ static void
check_fp2_mul3(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Add(bn_env, p, b, c, e);
  fp2_Multiply(bn_env, p, a, b, e);
  fp2_Multiply(bn_env, p, b, a, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_mul3


// mul4: e = a, f = 1, f = a*f
__global__ static void
check_fp2_mul4(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Copy(bn_env, p, a, e);
  fp2_Set(bn_env, p, f, 1U, 0U);
  fp2_Multiply(bn_env, p, a, f, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_mul4


// mul5: e = 0, f = a*e
__global__ static void
check_fp2_mul5(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Set(bn_env, p, e, 0U, 0U);
  fp2_Multiply(bn_env, p, a, e, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_mul5


// sqr1: e = a^2, f = a*a
__global__ static void
check_fp2_sqr1(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Square(bn_env, p, a, e);
  fp2_Multiply(bn_env, p, a, a, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_sqr1


// sqr2: e = 0, f = e^2
__global__ static void
check_fp2_sqr2(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Set(bn_env, p, e, 0U, 0U);
  fp2_Square(bn_env, p, e, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_sqr2


// inv1: e = 1, f = a^{-1}, f = f*a
__global__ static void
check_fp2_inv1(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Set(bn_env, p, e, 1U, 0U);
  fp2_Invert(bn_env, p, a, f);
  fp2_Multiply(bn_env, p, f, a, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_inv1


// inv2: b = a, e = a^{-1}, f = b^{-1}
__global__ static void
check_fp2_inv2(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Set(bn_env, p, a, b);
  fp2_Invert(bn_env, p, a, e);
  fp2_Invert(bn_env, p, b, f);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_inv2


// sqrt1: e = a^2, d = sqrt{e}, f = d^2 = e
__global__ static void
check_fp2_sqrt1(fp2_instance_t* device_instances,
    const uint32_t num_instances) {
  uint32_t instance_idx = (blockIdx.x*blockDim.x + threadIdx.x) / TPI;
  if (instance_idx >= num_instances) {
    return;
  }
  typedef cgbn_context_t<TPI> this_context_t;
  typedef cgbn_env_t<this_context_t, BITS> this_env_t;
  this_context_t bn_context(cgbn_monitor_t::cgbn_no_checks, nullptr, num_instances);
  this_env_t bn_env(bn_context.env<this_env_t>());
  typename this_env_t::cgbn_t p;
  cgbn2_t<this_env_t> a, b, c, d, e, f;
  cgbn_load(bn_env, p, &(device_instances[instance_idx].p));
  cgfp2_load(bn_env, a, &(device_instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(device_instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(device_instances[instance_idx].c), p);
  cgfp2_load(bn_env, d, &(device_instances[instance_idx].d), p);
  cgfp2_load(bn_env, e, &(device_instances[instance_idx].e), p);
  cgfp2_load(bn_env, f, &(device_instances[instance_idx].f), p);
  fp2_Square(bn_env, p, a, e);
  fp2_Sqrt(bn_env, p, e, d);
  fp2_Square(bn_env, p, d, f);
  cgfp2_store(bn_env, &(device_instances[instance_idx].a), a, p);
  cgfp2_store(bn_env, &(device_instances[instance_idx].b), b, p);
  cgfp2_store(bn_env, &(device_instances[instance_idx].c), c, p);
  cgfp2_store(bn_env, &(device_instances[instance_idx].d), d, p);
  cgfp2_store(bn_env, &(device_instances[instance_idx].e), e, p);
  cgfp2_store(bn_env, &(device_instances[instance_idx].f), f, p);
  device_instances[instance_idx].is_equal = fp2_IsEqual(bn_env, p, e, f);
} // check_fp2_sqrt1


__host__ static bool
test_fp2(const sike_params_raw_t& params_raw,
    const uint32_t num_instances) {
  CUDA_CHECK(cudaSetDevice(0));
  // launch with 32 threads per instance, 128 threads (4 instances) per block
  const uint32_t size_of_instances = sizeof(fp2_instance_t) * num_instances;
  const uint32_t ngrid = (num_instances + 3) / 4;
  bool rc = false;
  // add1: e = (a+b)+c, f = a+(b+c)
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_add1<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // add2: e = a+b, f = b+a
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_add2<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // add3: e = a, f = 0, f = a+f
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_add3<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // add4: e = 0, f = -a, f = f+a
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_add4<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // sub1: e = (a-b)-c, f = a-(b+c)
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_sub1<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // sub2: e = a-b, f = b-a, f = -f
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_sub2<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // sub3: e = a, f = 0, f = a-f
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_sub3<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // sub4: e = 0, f = -a, f = a+f
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_sub4<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // mul1: e = (a*b)*c, f = a*(b*c)
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_mul1<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // mul2: e = a*(b+c), f = a*b+a*c
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_mul2<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // mul3: e = a*b, f = b*a
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_mul3<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // mul4: e = a, d = 1, d = a*d
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_mul4<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // mul5: e = 0, f = a*e
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_mul5<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // sqr1: e = a^2, f = a*a
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_sqr1<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // sqr2: e = 0, f = e^2
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_sqr2<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // inv1: e = 1, f = a^{-1}, f = f*a
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_inv1<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // inv2: b = a, e = a^{-1}, f = b^{-1}
  if (true) {
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_inv2<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }
  // sqrt1: b = a^2, e = sqrt{b}, e = e^2, f = sqrt{b}, f = f^2
  if (true) {
    const bool debug = true;
    fp2_instance_t* host_instances = new fp2_instance_t[num_instances];
    //ASSUME(host_instances != nullptr);
    fp2_instance_t* device_instances = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&device_instances, size_of_instances));
    fp2_generate_instances(host_instances, num_instances, params_raw.p);
    CUDA_CHECK(cudaMemcpy(device_instances, host_instances, size_of_instances, cudaMemcpyHostToDevice));
    check_fp2_sqrt1<<<ngrid, 128>>>(device_instances, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_instances, device_instances, size_of_instances, cudaMemcpyDeviceToHost));
    if (!debug) {
      const uint32_t nlimbs = (BITS+31) / 32;
      const uint32_t i = 0;
      println_value_("p    ", host_instances[i].p._limbs, nlimbs);
      println_value_("a.x0 ", host_instances[i].a.x0._limbs, nlimbs);
      println_value_("a.x1 ", host_instances[i].a.x1._limbs, nlimbs);
      println_value_("d.x0 ", host_instances[i].d.x0._limbs, nlimbs);
      println_value_("d.x1 ", host_instances[i].d.x1._limbs, nlimbs);
      println_value_("e.x0 ", host_instances[i].e.x0._limbs, nlimbs);
      println_value_("e.x1 ", host_instances[i].e.x1._limbs, nlimbs);
      println_value_("f.x0 ", host_instances[i].f.x0._limbs, nlimbs);
      println_value_("f.x1 ", host_instances[i].f.x1._limbs, nlimbs);
    }
    rc = fp2_is_equal(host_instances, num_instances);
    fp2_free_instances(host_instances, num_instances);
    CUDA_CHECK(cudaFree(device_instances));
    if (rc == false) {
      return false;
    }
  }

  return true;
} // test_fp2
#elif 0
__host__ static bool
test_fp2(const sike_params_raw_t& params_raw,
    const uint32_t num_instances) {
  return true;
}
#else
  #error Not defined: test_fp2()
#endif // test for GF(p^2)-arithmetic operations


// This function is called by test.cu.
bool
test_arith(const sike_params_raw_t& params_raw) {
#if 1
  const uint32_t num_instances = 2; //32*32;
  const uint32_t num_loops = 1;	    //32;
#else
  const uint32_t num_instances = 4;
  const uint32_t num_loops = 1;  
#endif

  // Check GF(p) operations
  for (uint32_t i = 0; i < num_loops; ++i) {
    bool rc_fp = test_fp(params_raw, num_instances);
    if (rc_fp == false) { 
      return false; 
    }
  }

  // Check GF(p^2) operations
  for (uint32_t i = 0; i < num_loops; ++i) {
    bool rc_fp2 = test_fp2(params_raw, num_instances);
    if (rc_fp2 == false) { 
      return false; 
    }
  }

  return true;
}


// ================================================
#if 0 // original code
#define TEST_LOOPS 10

static int test_fp(const sike_params_t *params) {
  int rc = 0, n;

  const ff_Params *p = params->EA.ffData;

  mp a, b, c, d, e, f;

  fp_Init(p, a);
  fp_Init(p, b);
  fp_Init(p, c);
  fp_Init(p, d);
  fp_Init(p, e);
  fp_Init(p, f);


  printf(
    "\n--------------------------------------------------------------------------------------------------------\n\n");
  printf("Testing field arithmetic over GF(p): \n\n");

  // Field addition over the prime p751
  rc = 0;
  for (n = 0; n < TEST_LOOPS; n++) {
    fp_Rand(p, a);
    fp_Rand(p, b);
    fp_Rand(p, c);
    fp_Rand(p, d);
    fp_Rand(p, e);
    fp_Rand(p, f);

    fp_Add(p, a, b, d);
    fp_Add(p, d, c, e);                 // e = (a+b)+c
    fp_Add(p, b, c, d);
    fp_Add(p, d, a, f);                 // f = a+(b+c)
    if (!fp_IsEqual(p, e, f)) {
      rc = 1;
      break;
    }

    fp_Add(p, a, b, d);                                     // d = a+b
    fp_Add(p, b, a, e);                                     // e = b+a
    if (!fp_IsEqual(p, d, e)) {
      rc = 1;
      break;
    }

    fp_Zero(p, b);
    fp_Add(p, a, b, d);                                     // d = a+0
    if (!fp_IsEqual(p, a, d)) {
      rc = 1;
      break;
    }

    fp_Zero(p, b);
    fp_Copy(p, a, d);
    fp_Negative(p, d, d);
    fp_Add(p, a, d, e);                                     // e = a+(-a)
    if (!fp_IsEqual(p, e, b)) {
      rc = 1;
      break;
    }
  }
  if (!rc) printf("  GF(p) addition tests ............................................ PASSED");
  else {
    printf("  GF(p) addition tests... FAILED");
    printf("\n");
    goto end;
  }
  printf("\n");

  // Field subtraction over the prime p751
  rc = 0;
  for (n = 0; n < TEST_LOOPS; n++) {
    fp_Rand(p, a);
    fp_Rand(p, b);
    fp_Rand(p, c);
    fp_Rand(p, d);
    fp_Rand(p, e);
    fp_Rand(p, f);

    fp_Subtract(p, a, b, d);
    fp_Subtract(p, d, c, e);                 // e = (a-b)-c
    fp_Add(p, b, c, d);
    fp_Subtract(p, a, d, f);                 // f = a-(b+c)
    if (!fp_IsEqual(p, e, f)) {
      rc = 1;
      break;
    }

    fp_Subtract(p, a, b, d);                                     // d = a-b
    fp_Subtract(p, b, a, e);
    fp_Negative(p, e, e);                                           // e = -(b-a)
    if (!fp_IsEqual(p, d, e)) {
      rc = 1;
      break;
    }

    fp_Zero(p, b);
    fp_Subtract(p, a, b, d);                                     // d = a-0
    if (!fp_IsEqual(p, a, d)) {
      rc = 1;
      break;
    }

    fp_Zero(p, b);
    fp_Copy(p, a, d);
    fp_Subtract(p, a, d, e);                                     // e = a+(-a)
    if (!fp_IsEqual(p, e, b)) {
      rc = 1;
      break;
    }
  }
  if (!rc) printf("  GF(p) subtraction tests ......................................... PASSED");
  else {
    printf("  GF(p) subtraction tests... FAILED");
    printf("\n");
    goto end;
  }
  printf("\n");

  // Field multiplication over the prime p751
  rc = 0;
  for (n = 0; n < TEST_LOOPS; n++) {
    fp_Rand(p, a);
    fp_Rand(p, b);
    fp_Rand(p, c);

    fp_Multiply(p, a, b, d);
    fp_Multiply(p, d, c, e);                          // e = (a*b)*c
    fp_Multiply(p, b, c, d);
    fp_Multiply(p, d, a, f);                          // f = a*(b*c)
    if (!fp_IsEqual(p, e, f)) {
      rc = 1;
      break;
    }

    fp_Add(p, b, c, d);
    fp_Multiply(p, a, d, e);                               // e = a*(b+c)
    fp_Multiply(p, a, b, d);
    fp_Multiply(p, a, c, f);
    fp_Add(p, d, f, f);    // f = a*b+a*c
    if (!fp_IsEqual(p, e, f)) {
      rc = 1;
      break;
    }

    fp_Multiply(p, a, b, d);                                                      // d = a*b
    fp_Multiply(p, b, a, e);                                                      // e = b*a
    if (!fp_IsEqual(p, d, e)) {
      rc = 1;
      break;
    }

    fp_Constant(p, 1, b);
    fp_Multiply(p, a, b, d);                                                      // d = a*1
    if (!fp_IsEqual(p, a, d)) {
      rc = 1;
      break;
    }

    fp_Zero(p, b);
    fp_Multiply(p, a, b, d);                                                      // d = a*0
    if (!fp_IsEqual(p, b, d)) {
      rc = 1;
      break;
    }
  }
  if (!rc) printf("  GF(p) multiplication tests ...................................... PASSED");
  else {
    printf("  GF(p) multiplication tests... FAILED");
    printf("\n");
    goto end;
  }
  printf("\n");

  // Field squaring over the prime p751
  rc = 0;
  for (n = 0; n < TEST_LOOPS; n++) {
    fp_Rand(p, a);

    fp_Square(p, a, b);                                 // b = a^2
    fp_Multiply(p, a, a, c);                             // c = a*a
    if (!fp_IsEqual(p, b, c)) {
      rc = 1;
      break;
    }

    fp_Zero(p, a);
    fp_Square(p, a, d);                                 // d = 0^2
    if (!fp_IsEqual(p, a, d)) {
      rc = 1;
      break;
    }
  }
  if (!rc) printf("  GF(p) squaring tests............................................. PASSED");
  else {
    printf("  GF(p) squaring tests... FAILED");
    printf("\n");
    goto end;
  }
  printf("\n");

  // Field inversion over the prime p751
  rc = 0;
  for (n = 0; n < TEST_LOOPS; n++) {
    fp_Rand(p, a);
    fp_Constant(p, 1, d);
    fp_Copy(p, a, b);
    fp_Invert(p, a, a);
    fp_Multiply(p, a, b, c);                             // c = a*a^-1
    if (!fp_IsEqual(p, c, d)) {
      rc = 1;
      break;
    }

    fp_Rand(p, a);
    fp_Copy(p, a, b);
    fp_Invert(p, b, b);                                     // a = a^-1 by exponentiation
    fp_Invert(p, b, b);                                     // a = a^-1 by exponentiation
    if (!fp_IsEqual(p, a, b)) {
      rc = 1;
      break;
    }
  }
  if (!rc) printf("  GF(p) inversion tests............................................ PASSED");
  else {
    printf("  GF(p) inversion tests... FAILED");
    printf("\n");
    goto end;
  }
  printf("\n");


end:
  fp_Clear(p, a);
  fp_Clear(p, b);
  fp_Clear(p, c);
  fp_Clear(p, d);
  fp_Clear(p, e);
  fp_Clear(p, f);

  return rc;
}

static int test_fp2(const sike_params_t* params)
{ // Tests for the quadratic extension field arithmetic
  int rc = 0;

  const ff_Params *p = params->EA.ffData;

  int n;
  fp2 a, b, c, d, e, f, g, h, i, j;

  fp2_Init(p, &a);
  fp2_Init(p, &b);
  fp2_Init(p, &c);
  fp2_Init(p, &d);
  fp2_Init(p, &e);
  fp2_Init(p, &f);
  fp2_Init(p, &g);
  fp2_Init(p, &h);
  fp2_Init(p, &i);
  fp2_Init(p, &j);

  printf("\n--------------------------------------------------------------------------------------------------------\n\n");
  printf("Testing quadratic extension arithmetic over GF(p751^2): \n\n");

  // Addition over GF
  rc = 0;
  for (n=0; n<TEST_LOOPS; n++)
  {
    fp2_Rand(p, &a); fp2_Rand(p, &b); fp2_Rand(p, &c); fp2_Rand(p, &d); fp2_Rand(p, &e); fp2_Rand(p, &f);

    fp2_Add(p, &a, &b, &d); fp2_Add(p, &d, &c, &e);                 // e = (a+b)+c
    fp2_Add(p, &b, &c, &d); fp2_Add(p, &d, &a, &f);                 // f = a+(b+c)
    if (!fp2_IsEqual(p, &e,&f)) { rc = 1; break; }

    fp2_Add(p, &a, &b, &d);                                     // d = a+b
    fp2_Add(p, &b, &a, &e);                                     // e = b+a
    if (!fp2_IsEqual(p, &d,&e)) { rc = 1; break; }

    fp2_Set(p, &b, 0, 0);
    fp2_Add(p, &a, &b, &d);                                     // d = a+0
    if (!fp2_IsEqual(p, &a,&d)) { rc = 1; break; }

    fp2_Set(p, &b, 0, 0);
    fp2_Copy(p, &a, &d);
    fp2_Negative(p, &d, &d);
    fp2_Add(p, &a, &d, &e);                                     // e = a+(-a)
    if (!fp2_IsEqual(p, &e,&b)) { rc = 1; break; }
  }
  if (!rc) printf("  GF(p^2) addition tests .......................................... PASSED");
  else {
    printf("  GF(p^2) addition tests... FAILED"); printf("\n");
    goto end;
  }
  printf("\n");

  // Subtraction over GF
  rc = 0;
  for (n=0; n<TEST_LOOPS; n++)
  {
    fp2_Rand(p, &a); fp2_Rand(p, &b); fp2_Rand(p, &c); fp2_Rand(p, &d); fp2_Rand(p, &e); fp2_Rand(p, &f);

    fp2_Sub(p, &a, &b, &d); fp2_Sub(p, &d, &c, &e);                 // e = (a-b)-c
    fp2_Add(p,&b, &c, &d); fp2_Sub(p, &a, &d, &f);                 // f = a-(b+c)
    if (!fp2_IsEqual(p, &e,&f)) { rc = 1; break; }

    fp2_Sub(p, &a, &b, &d);                                     // d = a-b
    fp2_Sub(p, &b, &a, &e);
    fp2_Negative(p, &e, &e);                                           // e = -(b-a)
    if (!fp2_IsEqual(p, &d, &e)) { rc = 1; break; }

    fp2_Set(p, &b, 0, 0);
    fp2_Sub(p, &a, &b, &d);                                     // d = a-0
    if (!fp2_IsEqual(p, &a, &d)) { rc = 1; break; }

    fp2_Set(p, &b, 0, 0);
    fp2_Copy(p, &a, &d);
    fp2_Sub(p, &a, &d, &e);                                     // e = a+(-a)
    if (!fp2_IsEqual(p, &e, &b)) { rc = 1; break; }
  }
  if (!rc) printf("  GF(p^2) subtraction tests ....................................... PASSED");
  else { printf("  GF(p^2) subtraction tests... FAILED"); printf("\n"); goto end; }
  printf("\n");

  // Multiplication over GF
  rc = 0;
  for (n=0; n<TEST_LOOPS; n++)
  {
    fp2_Rand(p, &a); fp2_Rand(p, &b); fp2_Rand(p, &c);

    fp2_Multiply(p, &a, &b, &d); fp2_Multiply(p, &d, &c, &e);                          // e = (a*b)*c
    fp2_Multiply(p, &b, &c, &d); fp2_Multiply(p, &d, &a, &f);                          // f = a*(b*c)
    if (!fp2_IsEqual(p, &e, &f)) { rc = 1; break; }

    fp2_Add(p, &b, &c, &d); fp2_Multiply(p, &a, &d, &e);                               // e = a*(b+c)
    fp2_Multiply(p, &a, &b, &d); fp2_Multiply(p, &a, &c, &f); fp2_Add(p, &d, &f, &f);   // f = a*b+a*c
    if (!fp2_IsEqual(p, &e, &f)) { rc = 1; break; }

    fp2_Multiply(p, &a, &b, &d);                                                      // d = a*b
    fp2_Multiply(p, &b, &a, &e);                                                      // e = b*a
    if (!fp2_IsEqual(p, &d, &e)) { rc = 1; break; }

    fp2_Set(p, &b, 1, 0);
    fp2_Multiply(p, &a, &b, &d);                                                      // d = a*1
    if (!fp2_IsEqual(p, &a, &d)) { rc = 1; break; }

    fp2_Set(p, &b, 0, 0);
    fp2_Multiply(p, &a, &b, &d);                                                      // d = a*0
    if (!fp2_IsEqual(p, &b, &d)) { rc = 1; break; }
  }
  if (!rc) printf("  GF(p^2) multiplication tests .................................... PASSED");
  else { printf("  GF(p^2) multiplication tests... FAILED"); printf("\n"); goto end; }
  printf("\n");

  // Squaring over GF
  rc = 0;
  for (n=0; n<TEST_LOOPS; n++)
  {
    fp2_Rand(p, &a);

    fp2_Square(p, &a, &b);                                 // b = a^2
    fp2_Multiply(p, &a, &a, &c);                             // c = a*a
    if (!fp2_IsEqual(p, &b, &c)) { rc = 1; break; }

    fp2_Set(p, &a, 0, 0);
    fp2_Square(p, &a, &d);                                 // d = 0^2
    if (!fp2_IsEqual(p, &a, &d)) { rc = 1; break; }
  }
  if (!rc) printf("  GF(p^2) squaring tests........................................... PASSED");
  else { printf("  GF(p^2) squaring tests... FAILED"); printf("\n"); goto end; }
  printf("\n");

  // Inversion over GF
  rc = 0;
  for (n=0; n<TEST_LOOPS; n++)
  {
    fp2_Rand(p, &a);

    fp2_Set(p, &d, 1, 0);
    fp2_Copy(p, &a, &b);
    fp2_Invert(p, &a, &a);
    fp2_Multiply(p, &a, &b, &c);                             // c = a*a^-1
    if (!fp2_IsEqual(p, &c, &d)) { rc = 1; break; }

    fp2_Rand(p, &a);

    fp2_Copy(p, &a, &b);
    fp2_Invert(p, &a, &a);                                 // a = a^-1 with exponentiation
    fp2_Invert(p, &b, &b);                                 // a = a^-1 with exponentiation
    if (!fp2_IsEqual(p, &a, &b)) { rc = 1; break; }
  }
  if (!rc) printf("  GF(p^2) inversion tests.......................................... PASSED");
  else { printf("  GF(p^2) inversion tests... FAILED"); printf("\n"); goto end; }
  printf("\n");

  // Sqrt over GF
  rc = 0;
  for (n = 0; n < 100; ++n)
  {
    fp2_Rand(p, &a);
    fp2_Square(p, &a, &b);
    fp2_Sqrt(p, &b, &c, 0);
    fp2_Sqrt(p, &b, &d, 1);

    if (!fp2_IsEqual(p, &a, &c) && !fp2_IsEqual(p, &a, &d)) { rc = 1; break; }
  }
  if (!rc) printf("  GF(p^2) sqrt tests............................................... PASSED");
  else { printf("  GF(p^2) sqrt tests... FAILED"); printf("\n"); goto end; }
  printf("\n");

end:
  fp2_Clear(p, &a);
  fp2_Clear(p, &b);
  fp2_Clear(p, &c);
  fp2_Clear(p, &d);
  fp2_Clear(p, &e);
  fp2_Clear(p, &f);

  return rc;
}

int test_arith(const sike_params_t *params) {

  int rc = 0;
  rc = test_fp(params);
  if ( rc ) goto end;

  rc = test_fp2(params);
  if ( rc ) goto end;

end:
  return rc;
}
#endif // original code

// end of file
