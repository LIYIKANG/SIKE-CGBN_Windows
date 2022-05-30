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
#include <sys/time.h>
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
class fp2_time_params_t {
  public:
  // parameters used by the CGBN context
  static const uint32_t TPB=0;                     // get TPB from blockDim.x
  static const uint32_t MAX_ROTATION=4;            // good default value
  static const uint32_t SHM_LIMIT=0;               // no shared mem available
  static const bool     CONSTANT_TIME=false;       // constant time implementations aren't available yet
  // parameters used locally in the application
  static const uint32_t TPI=tpi;                   // threads per instance
  static const uint32_t BITS=bits;                 // instance size
}; // class fp2_time_params_t {


template<class params>
class fp2_time_t {
  public:

  // define the instance structure
  // a = a0 + a1*i
  typedef struct {
    cgbn_mem_t<params::BITS> x0; cgbn_mem_t<params::BITS> x1;
    cgbn_mem_t<params::BITS> a0; cgbn_mem_t<params::BITS> a1;
    cgbn_mem_t<params::BITS> e;
    cgbn_mem_t<params::BITS> p;
  } instance_t;

  typedef cgbn_context_t<params::TPI, params>   context_t;
  typedef cgbn_env_t<context_t, params::BITS>   env_t;
  typedef typename env_t::cgbn_t                bn_t;
  typedef typename env_t::cgbn_local_t          bn_local_t;

  context_t _context;
  env_t     _env;
  int32_t   _instance;

  __device__ __forceinline__ fp2_time_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance)
    : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {
      // Nothing to do
  }

  __device__ __forceinline__ void
  fp2_calc_thisThread(bn_t& x0, bn_t& x1, bn_t& a0, bn_t& a1, bn_t& e, bn_t& p) {
    // x = a^e
    if (true) { // over GF(p)
      cgbn_modular_power(_env, x0, a0, e, p);
    } else if (true) { // over GF(p^2)
      assert(false);
    }
  }

  __host__ static void
  fp2_time_cpu(instance_t* cpu, const uint32_t num_instances) {
    mpz_t x0, x1, a0, a1, e, p;
    mpz_inits(x0, x1, a0, a1, e, p, NULL);
    const uint32_t nlimbs = sizeof(cpu[0].p._limbs) / sizeof(cpu[0].p._limbs[0]);
    assert(nlimbs == params::BITS/32);
    for (uint32_t i = 0; i < num_instances; ++i) {
      to_mpz(x0, cpu[i].x0._limbs, nlimbs); to_mpz(x1, cpu[i].x1._limbs, nlimbs);      
      to_mpz(a0, cpu[i].a0._limbs, nlimbs); to_mpz(a1, cpu[i].a1._limbs, nlimbs);
      to_mpz(e, cpu[i].e._limbs, nlimbs);
      to_mpz(p, cpu[i].p._limbs, nlimbs);
      if (true) { // over GF(p)
        mpz_powm(x0, a0, e, p);
      } else if (true) { // over GF(p^2)
        assert(false);
      }
      from_mpz(x0, cpu[i].x0._limbs, nlimbs);from_mpz(x1, cpu[i].x1._limbs, nlimbs);
      from_mpz(a0, cpu[i].a0._limbs, nlimbs);from_mpz(a1, cpu[i].a1._limbs, nlimbs);
      from_mpz(e, cpu[i].e._limbs, nlimbs);
      from_mpz(p, cpu[i].p._limbs, nlimbs);
    }
    mpz_clears(x0, x1, a0, a1, e, p, NULL);
  }

  __host__ static instance_t*
  generate_instances(const uint32_t num_instances, const mpz_t p) {
    instance_t* instances = (instance_t*)malloc(sizeof(instance_t)*num_instances);
    for (uint32_t i = 0; i < num_instances; ++i) {
      fp_random(instances[i].x0._limbs, params::BITS/32, p);
      fp_random(instances[i].x1._limbs, params::BITS/32, p);
      fp_random(instances[i].a0._limbs, params::BITS/32, p);
      fp_random(instances[i].a1._limbs, params::BITS/32, p);
      fp_random(instances[i].e._limbs, params::BITS/32, p);
      from_mpz(p, instances[i].p._limbs, params::BITS/32);
    }
    return instances;
  }

  __host__ static bool
  is_same_results(const instance_t* cpu, const instance_t* gpu,  const uint32_t num_instances) {
    const uint32_t nlimbs = sizeof(cpu[0].a0._limbs) / sizeof(cpu[0].a0._limbs[0]);
    for (uint32_t i = 0; i < num_instances; ++i) {
      for (uint32_t j = 0; j < nlimbs; ++j) {
        if (cpu[i].x0._limbs[j] != gpu[i].x0._limbs[j]) {
          std::cerr << "cpu[" << i << "].x0._limbs[" << j << "]  " << cpu[i].x0._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].x0._limbs[" << j << "]  " << gpu[i].x0._limbs[j] << std::endl;
          return false;
        }
        if (cpu[i].x1._limbs[j] != gpu[i].x1._limbs[j]) {
          std::cerr << "cpu[" << i << "].x1._limbs[" << j << "]  " << cpu[i].x1._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].x1._limbs[" << j << "]  " << gpu[i].x1._limbs[j] << std::endl;
          return false;
        }

        if (cpu[i].a0._limbs[j] != gpu[i].a0._limbs[j]) {
          std::cerr << "cpu[" << i << "].a0._limbs[" << j << "]  " << cpu[i].a0._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].a0._limbs[" << j << "]  " << gpu[i].a0._limbs[j] << std::endl;
          return false;
        }
        if (cpu[i].a1._limbs[j] != gpu[i].a1._limbs[j]) {
          std::cerr << "cpu[" << i << "].a1._limbs[" << j << "]  " << cpu[i].a1._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].a1._limbs[" << j << "]  " << gpu[i].a1._limbs[j] << std::endl;
          return false;
        }

        if (cpu[i].e._limbs[j] != gpu[i].e._limbs[j]) {
          std::cerr << "cpu[" << i << "].e._limbs[" << j << "]  " << cpu[i].e._limbs[j] << std::endl;
          std::cerr << "gpu[" << i << "].e._limbs[" << j << "]  " << gpu[i].e._limbs[j] << std::endl;
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

}; // class fp2_time_t {



// kernel implementation using cgbn
//
// Unfortunately, the kernel must be separate from the fp2_time_t class
template<class params>
__global__ void
fp2_time_gpu(cgbn_error_report_t* report, typename fp2_time_t<params>::instance_t* gpumem, const uint32_t num_instances) {
  // decode an instance number from the blockIdx and threadIdx
  uint32_t thisIdx = (blockIdx.x*blockDim.x + threadIdx.x) / params::TPI;
  if (thisIdx >= num_instances) { return; }

  fp2_time_t<params> fpa(cgbn_report_monitor, report, thisIdx);
  typename fp2_time_t<params>::bn_t  a0, x0, e, p;
  typename fp2_time_t<params>::bn_t  a1, x1;

  cgbn_load(fpa._env, p, &(gpumem[thisIdx].p));
  cgbn_load(fpa._env, e, &(gpumem[thisIdx].e));
  cgfp2_load(fpa._env, x0, x1, &(gpumem[thisIdx].x0), &(gpumem[thisIdx].x1), p);
  cgfp2_load(fpa._env, a0, a1, &(gpumem[thisIdx].a0), &(gpumem[thisIdx].a1), p);

  fpa.fp2_calc_thisThread(x0, x1, a0, a1, e, p);

  cgfp2_store(fpa._env, &(gpumem[thisIdx].x0), &(gpumem[thisIdx].x1), x0, x1, p);
  cgfp2_store(fpa._env, &(gpumem[thisIdx].a0), &(gpumem[thisIdx].a1), a0, a1, p);
  cgbn_store(fpa._env, &(gpumem[thisIdx].e), e);
  cgbn_store(fpa._env, &(gpumem[thisIdx].p), p);
}


template<class params>
void run_test(const uint32_t num_instances, const mpz_t p) {
  typedef typename fp2_time_t<params>::instance_t instance_t;

  const uint32_t TPB = (params::TPB == 0) ? 128 : params::TPB; // default threads per block to 128
  const uint32_t TPI = params::TPI;
  const uint32_t IPB = TPB/TPI; // IPB is instances per block

  // time measuer 
  cudaEvent_t gpu_start, gpu_stop;
  float gpu_time;

  // create a cgbn_error_report for CGBN to report back errors
  cgbn_error_report_t* report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  std::cout << "Genereate instances ..." << std::endl;
  instance_t* org = fp2_time_t<params>::generate_instances(num_instances, p);
  const uint32_t total_instance_sizeinbyte = sizeof(instance_t) * num_instances;
  instance_t* cpu = (instance_t*)malloc(total_instance_sizeinbyte);
  instance_t* gpu = (instance_t*)malloc(total_instance_sizeinbyte);
  memcpy(cpu, org, total_instance_sizeinbyte);
  memcpy(gpu, org, total_instance_sizeinbyte);
  
  std::cout << "Copy instances ..." << std::endl;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaEventCreate(&gpu_start));
  CUDA_CHECK(cudaEventCreate(&gpu_stop));
  
  instance_t* gpumem = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&gpumem, total_instance_sizeinbyte));
  CUDA_CHECK(cudaMemcpy(gpumem, gpu, total_instance_sizeinbyte, cudaMemcpyHostToDevice));
  std::cout << "Launch GPU and CPU ..." << std::endl;
  if (true) {
    // launch kernel with blocks=ceil(instance_num_instances/IPB) and threads=TPB
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(gpu_start, 0));
    fp2_time_gpu<params><<<(num_instances+IPB-1)/IPB, TPB>>>(report, gpumem, num_instances);
    CUDA_CHECK(cudaEventRecord(gpu_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(gpu_stop));
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop));
    std::cout << "GPU " << gpu_time << " ms" << std::endl;
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
  }
  if (true) {
    struct timeval cpu_start;
    struct timeval cpu_stop;
    gettimeofday(&cpu_start, NULL);
    fp2_time_t<params>::fp2_time_cpu(cpu, num_instances);
    gettimeofday(&cpu_stop, NULL);    
    double cpu_start_ms = cpu_start.tv_sec * 1000.0 + cpu_start.tv_usec / 1000.0;
    double cpu_stop_ms = cpu_stop.tv_sec * 1000.0 + cpu_stop.tv_usec / 1000.0;
    std::cout << "CPU " << (cpu_stop_ms - cpu_start_ms) << " ms" << std::endl;
  }
  
  
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(gpu, gpumem, total_instance_sizeinbyte, cudaMemcpyDeviceToHost));

  std::cout << "Compare CPU results with GPU results ..." << std::endl;
  bool is_ok = fp2_time_t<params>::is_same_results(cpu, gpu, num_instances);
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
  // class fp2_time_params_t {
  const uint32_t tpi = 8;
  const uint32_t bits = ((p_sizeinbit + 31) / 32) * 32;
  typedef fp2_time_params_t<tpi, bits> params;

  run_test<params>(1024*1024, p);

  mpz_clear(p);
  return 0;
}

// end of file
