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
#include <string.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/support.h"

// For this example, there are quite a few template parameters that are used to generate the actual code.
// In order to simplify passing many parameters, we use the same approach as the CGBN library, which is to
// create a container class with static constants and then pass the class.

// The CGBN context uses the following three parameters:
//   TBP             - threads per block (zero means to use the blockDim.x)
//   MAX_ROTATION    - must be small power of 2, imperically, 4 works well
//   SHM_LIMIT       - number of bytes of dynamic shared memory available to the kernel
//   CONSTANT_TIME   - require constant time algorithms (currently, constant time algorithms are not available)

// Locally it will also be helpful to have several parameters:
//   TPI             - threads per instance
//   BITS            - number of bits per instance
//   WINDOW_BITS     - number of bits to use for the windowed exponentiation

template<uint32_t tpi, uint32_t bits>
class fp_arith_params_t {
  public:
  // parameters used by the CGBN context
  static const uint32_t TPB=0;                     // get TPB from blockDim.x
  static const uint32_t MAX_ROTATION=4;            // good default value
  static const uint32_t SHM_LIMIT=0;               // no shared mem available
  static const bool     CONSTANT_TIME=false;       // constant time implementations aren't available yet
  // parameters used locally in the application
  static const uint32_t TPI=tpi;                   // threads per instance
  static const uint32_t BITS=bits;                 // instance size
}; // class fp_arith_params_t {


template<class params>
class fp_arith_t {
  public:

  // define the instance structure
  typedef struct {
    cgbn_mem_t<params::BITS> a;
    cgbn_mem_t<params::BITS> b;
    cgbn_mem_t<params::BITS> c; // a+b
    cgbn_mem_t<params::BITS> d; // a-b
    cgbn_mem_t<params::BITS> e; // a*b
    cgbn_mem_t<params::BITS> f; // a/b
    cgbn_mem_t<params::BITS> g; // a*a
    cgbn_mem_t<params::BITS> h; // a^b
    cgbn_mem_t<params::BITS> p;
  } instance_t;

  typedef cgbn_context_t<params::TPI, params>   context_t;
  typedef cgbn_env_t<context_t, params::BITS>   env_t;
  typedef typename env_t::cgbn_t                bn_t;
  typedef typename env_t::cgbn_local_t          bn_local_t;

  context_t _context;
  env_t     _env;
  int32_t   _instance;

  __device__ __forceinline__ fp_arith_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance)
    : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {
      // Nothing to do
  }

  __device__ __forceinline__ void
  fp_calc_thisThread(bn_t& a, bn_t& b, bn_t& c, bn_t& d, bn_t& e, bn_t& f, bn_t& g, bn_t& h, bn_t& p) {
    // a+b
    cgfp_add(_env, c, a, b, p);
    // a-b
    cgfp_sub(_env, d, a, b, p);
    // a*b
    cgfp_mul(_env, e, a, b, p, MONT_RED);
    // a/b
    cgfp_div(_env, f, a, b, p);
    // a*a
    cgfp_sqr(_env, g, a, p);
    // a^b
    cgfp_modular_power(_env, h, a, b, p);
  }


  __host__ static void
  fp_arith_cpu(instance_t* cpu, const uint32_t num_instances) {
    mpz_t a; mpz_init(a);
    mpz_t b; mpz_init(b);
    mpz_t c; mpz_init(c); // a+b
    mpz_t d; mpz_init(d); // a-b
    mpz_t e; mpz_init(e); // a*b
    mpz_t f; mpz_init(f); // a/b
    mpz_t g; mpz_init(g); // a*a
    mpz_t h; mpz_init(h); // a^b
    mpz_t p; mpz_init(p);

    const uint32_t nlimbs = sizeof(cpu[0].a._limbs) / sizeof(cpu[0].a._limbs[0]);
    assert(nlimbs == params::BITS/32);
    for (uint32_t i = 0; i < num_instances; ++i) {
      to_mpz(a, cpu[i].a._limbs, nlimbs);
      to_mpz(b, cpu[i].b._limbs, nlimbs);
      to_mpz(p, cpu[i].p._limbs, nlimbs);
      /* a+b */ mpz_add(c, a, b); mpz_mod(c, c, p);
      /* a-b */ mpz_sub(d, a, b); mpz_mod(d, d, p); if (mpz_cmp_ui(d, 0U) < 0) { mpz_add(d, d, p); }
      /* a*b */ mpz_mul(e, a, b); mpz_mod(e, e, p);
      /* a/b */ mpz_invert(f, b, p); mpz_mul(f, a, f); mpz_mod(f, f, p);
      /* a*a */ mpz_mul(g, a, a); mpz_mod(g, g, p);
      /* a^b */ mpz_powm (h, a, b, p);
      from_mpz(a, cpu[i].a._limbs, nlimbs);
      from_mpz(b, cpu[i].b._limbs, nlimbs);
      from_mpz(c, cpu[i].c._limbs, nlimbs);
      from_mpz(d, cpu[i].d._limbs, nlimbs);
      from_mpz(e, cpu[i].e._limbs, nlimbs);
      from_mpz(f, cpu[i].f._limbs, nlimbs);
      from_mpz(g, cpu[i].g._limbs, nlimbs);
      from_mpz(h, cpu[i].h._limbs, nlimbs);
      from_mpz(p, cpu[i].p._limbs, nlimbs);
    }

    mpz_clear(a); mpz_clear(b); mpz_clear(c); mpz_clear(d);
    mpz_clear(e); mpz_clear(f); mpz_clear(g); mpz_clear(h);
    mpz_clear(p);
  }

  __host__ static instance_t*
  generate_instances(const uint32_t num_instances, const mpz_t p) {
    instance_t* instances = (instance_t *)malloc(sizeof(instance_t)*num_instances);
    for (uint32_t i = 0; i < num_instances; ++i) {
      fp_random(instances[i].a._limbs, params::BITS/32, p);
      fp_random(instances[i].b._limbs, params::BITS/32, p);
      from_mpz(p, instances[i].p._limbs, params::BITS/32);
    }
    return instances;
  }

  __host__ static bool
  is_same_results(const instance_t* cpu, const instance_t* gpu,  const uint32_t num_instances) {
    const uint32_t nlimbs = sizeof(cpu[0].a._limbs) / sizeof(cpu[0].a._limbs[0]);
    for (uint32_t i = 0; i < num_instances; ++i) {
      for (uint32_t j = 0; j < nlimbs; ++j) {
        if (cpu[i].a._limbs[j] != gpu[i].a._limbs[j]) {
          std::cerr << "cpu[" << i << "].a._limbs[" << j << "]  " << cpu[i].a._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].a._limbs[" << j << "]  " << gpu[i].a._limbs[j] << std::endl;
          return false;
        }
        if (cpu[i].b._limbs[j] != gpu[i].b._limbs[j]) {
          std::cerr << "cpu[" << i << "].b._limbs[" << j << "]  " << cpu[i].b._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].b._limbs[" << j << "]  " << gpu[i].b._limbs[j] << std::endl;
          return false;
        }
        if (cpu[i].c._limbs[j] != gpu[i].c._limbs[j]) {
          std::cerr << "cpu[" << i << "].c._limbs[" << j << "]  " << cpu[i].c._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].c._limbs[" << j << "]  " << gpu[i].c._limbs[j] << std::endl;
          return false;
        }
        if (cpu[i].d._limbs[j] != gpu[i].d._limbs[j]) {
          std::cerr << "cpu[" << i << "].d._limbs[" << j << "]  " << cpu[i].d._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].d._limbs[" << j << "]  " << gpu[i].d._limbs[j] << std::endl;
          return false;
        }
        if (cpu[i].e._limbs[j] != gpu[i].e._limbs[j]) {
          std::cerr << "cpu[" << i << "].e._limbs[" << j << "]  " << cpu[i].e._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].e._limbs[" << j << "]  " << gpu[i].e._limbs[j] << std::endl;
          return false;
        }
        if (cpu[i].f._limbs[j] != gpu[i].f._limbs[j]) {
          std::cerr << "cpu[" << i << "].f._limbs[" << j << "]  " << cpu[i].f._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].f._limbs[" << j << "]  " << gpu[i].f._limbs[j] << std::endl;
          return false;
        }
        if (cpu[i].g._limbs[j] != gpu[i].g._limbs[j]) {
          std::cerr << "cpu[" << i << "].g._limbs[" << j << "]  " << cpu[i].g._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].g._limbs[" << j << "]  " << gpu[i].g._limbs[j] << std::endl;
          return false;
        }
        if (cpu[i].h._limbs[j] != gpu[i].h._limbs[j]) {
          std::cerr << "cpu[" << i << "].h._limbs[" << j << "]  " << cpu[i].h._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].h._limbs[" << j << "]  " << gpu[i].h._limbs[j] << std::endl;
          return false;
        }
        if (cpu[i].p._limbs[j] != gpu[i].p._limbs[j]) {
          std::cerr << "cpu[" << i << "].p._limbs[" << j << "]  " << cpu[i].p._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].p._limbs[" << j << "]  " << gpu[i].p._limbs[j] << std::endl;
          return false;
        }
      }
    }
    return true;
  }

}; // class fp_arith_t {



// kernel implementation using cgbn
//
// Unfortunately, the kernel must be separate from the fp_arith_t class
template<class params>
__global__ void
fp_arith_gpu(cgbn_error_report_t* report, typename fp_arith_t<params>::instance_t* gpumem, const uint32_t num_instances) {
  // decode an instance number from the blockIdx and threadIdx
  uint32_t thisIdx = (blockIdx.x*blockDim.x + threadIdx.x) / params::TPI;
  if (thisIdx >= num_instances) { return; }

  fp_arith_t<params> fpa(cgbn_report_monitor, report, thisIdx);
  typename fp_arith_t<params>::bn_t  a, b, c, d, e, f, g, h, p;

  // the loads and stores can go in the class, but it seems more natural to have them
  // here and to pass in and out bignums
  cgbn_load(fpa._env, p, &(gpumem[thisIdx].p));
  cgfp_load(fpa._env, a, &(gpumem[thisIdx].a), p);
  cgfp_load(fpa._env, b, &(gpumem[thisIdx].b), p);

  cgfp_set_ui32(fpa._env, c, 0U, p);
  cgfp_set_ui32(fpa._env, d, 0U, p);
  cgfp_set_ui32(fpa._env, e, 0U, p);
  cgfp_set_ui32(fpa._env, f, 0U, p);
  cgfp_set_ui32(fpa._env, g, 0U, p);
  cgfp_set_ui32(fpa._env, h, 0U, p);

  // this can be either fixed_window_fp_arith or sliding_window_fp_arith.
  // when TPI<32, fixed window runs much faster because it is less divergent, so we use it here
  fpa.fp_calc_thisThread(a, b, c, d, e, f, g, h, p);

  cgfp_store(fpa._env, &(gpumem[thisIdx].a), a, p);
  cgfp_store(fpa._env, &(gpumem[thisIdx].b), b, p);
  cgfp_store(fpa._env, &(gpumem[thisIdx].c), c, p);
  cgfp_store(fpa._env, &(gpumem[thisIdx].d), d, p);
  cgfp_store(fpa._env, &(gpumem[thisIdx].e), e, p);
  cgfp_store(fpa._env, &(gpumem[thisIdx].f), f, p);
  cgfp_store(fpa._env, &(gpumem[thisIdx].g), g, p);
  cgfp_store(fpa._env, &(gpumem[thisIdx].h), h, p);
  cgbn_store(fpa._env, &(gpumem[thisIdx].p), p);
}


template<class params>
void run_test(const uint32_t num_instances, const mpz_t p) {
  typedef typename fp_arith_t<params>::instance_t instance_t;

  const uint32_t TPB = (params::TPB == 0) ? 128 : params::TPB; // default threads per block to 128
  const uint32_t TPI = params::TPI;
  const uint32_t IPB = TPB/TPI; // IPB is instances per block

  // create a cgbn_error_report for CGBN to report back errors
  cgbn_error_report_t* report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  std::cout << "Genereate instances ..." << std::endl;
  instance_t* org = fp_arith_t<params>::generate_instances(num_instances, p);
  const uint32_t total_instance_sizeinbyte = sizeof(instance_t) * num_instances;
  instance_t* cpu = (instance_t*)malloc(total_instance_sizeinbyte);
  instance_t* gpu = (instance_t*)malloc(total_instance_sizeinbyte);
  memcpy(cpu, org, total_instance_sizeinbyte);
  memcpy(gpu, org, total_instance_sizeinbyte);

  std::cout << "Copy instances ..." << std::endl;
  CUDA_CHECK(cudaSetDevice(0));
  instance_t* gpumem = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&gpumem, total_instance_sizeinbyte));
  CUDA_CHECK(cudaMemcpy(gpumem, gpu, total_instance_sizeinbyte, cudaMemcpyHostToDevice));
  std::cout << "Launch GPU and CPU ..." << std::endl;
  if (true) {
    // launch kernel with blocks=ceil(instance_num_instances/IPB) and threads=TPB
    fp_arith_gpu<params><<<(num_instances+IPB-1)/IPB, TPB>>>(report, gpumem, num_instances);
  }
  if (true) {
    fp_arith_t<params>::fp_arith_cpu(cpu, num_instances);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(gpu, gpumem, total_instance_sizeinbyte, cudaMemcpyDeviceToHost));

  std::cout << "Compare CPU results with GPU results ..." << std::endl;
  bool is_ok = fp_arith_t<params>::is_same_results(cpu, gpu, num_instances);
  if (is_ok == false) { exit(EXIT_FAILURE); }

  // clean up
  CGBN_CHECK(report);
  CUDA_CHECK(cgbn_error_report_free(report));
  CUDA_CHECK(cudaFree(gpumem));
  free(org);
  free(cpu);
  free(gpu);
}

int main() {
  // 230-bit prime
  const char* p_str = "963068464539673160758804110401004374293008876561574715965035148628171";
  const uint32_t p_sizeinbit = 230;
  mpz_t p; mpz_init(p);
  mpz_set_str(p, p_str, 10);
  assert(p_sizeinbit == mpz_sizeinbase(p, 2));

  // template<uint32_t tpi, uint32_t bits>
  // class fp_arith_params_t {
  const uint32_t tpi = 8;
  const uint32_t bits = ((p_sizeinbit + 31) / 32) * 32;
  typedef fp_arith_params_t<tpi, bits> params;

  run_test<params>(16, p);

  mpz_clear(p);
  return 0;
}

// end of file
