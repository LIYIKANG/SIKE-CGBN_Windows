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



// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 32
#define NLIMBS ((434+31)/32)
#define BITS (32*NLIMBS)
#define NUM_INSTANCES (1024*4)


typedef enum {
  add_ope = 0, sub_ope, mul_ope, div_ope, madd_ope,
} operation_t;
#if !defined(OPE)
#define OPE madd_ope
#endif // !defined(OPE)

const char* p434 = "0x2341F271773446CFC5FD681C520567BC65C783158AEA3FDC1767AE2FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";

// Declare the instance type
typedef struct {
  cgbn2_mem_t<BITS> a;
  cgbn2_mem_t<BITS> b;
  cgbn2_mem_t<BITS> c;
  cgbn2_mem_t<BITS> z;
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
    if (true) { 
      fp_random(instances[i].a.x._limbs, nlimbs, p);
      fp_random(instances[i].a.y._limbs, nlimbs, p);
      fp_random(instances[i].b.x._limbs, nlimbs, p);
      fp_random(instances[i].b.y._limbs, nlimbs, p);
      fp_random(instances[i].c.x._limbs, nlimbs, p);
      fp_random(instances[i].c.y._limbs, nlimbs, p);
    } else {
      std::cout << "****** Bad random ******  Luke" << std::endl;
      for (uint32_t j = 0; j < nlimbs; ++j) {
        instances[i].a.x._limbs[j] = (j == 0) ? rand() % 9 + 1 : 0;
        instances[i].a.y._limbs[j] = (j == 0) ? rand() % 9 + 1 : 0;
        instances[i].b.x._limbs[j] = (j == 0) ? rand() % 9 + 1 : 0;
        instances[i].b.y._limbs[j] = (j == 0) ? rand() % 9 + 1 : 0;
        instances[i].c.x._limbs[j] = (j == 0) ? rand() % 9 + 1 : 0;
        instances[i].c.y._limbs[j] = (j == 0) ? rand() % 9 + 1 : 0;
      }
    }
    from_mpz(p, instances[i].p._limbs, nlimbs);
  }
  mpz_clear(p);
  // clear the random state.
  fp_random(nullptr, nlimbs, p);
  return instances;
} // generate_instances


// support routine to verify the GPU results using the CPU
bool
verify_results(instance_t* instances, const uint32_t num_instances) {
  // return code: true: success, false: failure
  bool rc = true; 
  
  uint32_t** correctx = (uint32_t**)malloc(sizeof(uint32_t*) * num_instances);
  uint32_t** correcty = (uint32_t**)malloc(sizeof(uint32_t*) * num_instances);
  assert(correctx != NULL && correcty != NULL);
  for (uint32_t i = 0; i < num_instances; ++i) {
    correctx[i] = (uint32_t*)malloc(sizeof(uint32_t) * NLIMBS);
    correcty[i] = (uint32_t*)malloc(sizeof(uint32_t) * NLIMBS);
    assert(correctx[i] != NULL && correcty[i] != NULL);
  }

  // Execute the same computation using GMP.
  mpz_t ax,ay, bx,by, cx,cy,  dx,dy,  ex,ey, fx,fy, zx,zy, p;
  mpz_inits(ax,ay, bx,by, cx,cy,  dx,dy,  ex,ey, fx,fy, zx,zy, p, NULL);
  mpz_set_str(p, p434, 0);
  for (uint32_t i = 0; i < num_instances; ++i) {
    to_mpz(ax, instances[i].a.x._limbs, NLIMBS);
    to_mpz(ay, instances[i].a.y._limbs, NLIMBS);
    to_mpz(bx, instances[i].b.x._limbs, NLIMBS);
    to_mpz(by, instances[i].b.y._limbs, NLIMBS);
    to_mpz(cx, instances[i].c.x._limbs, NLIMBS);
    to_mpz(cy, instances[i].c.y._limbs, NLIMBS);
    if (OPE == add_ope) { // z = a+b
      mpz_add(zx, ax, bx); mpz_mod(zx, zx, p);
      mpz_add(zy, ay, by); mpz_mod(zy, zy, p);
    } else if (OPE == sub_ope) { // z = a-b
      mpz_sub(zx, ax, bx); mpz_mod(zx, zx, p);
      if (mpz_cmp_ui(zx, 0U) < 0) { mpz_add(zx, zx, p); }
      mpz_sub(zy, ay, by); mpz_mod(zy, zy, p);
      if (mpz_cmp_ui(zy, 0U) < 0) { mpz_add(zy, zy, p); }
    } else if (OPE == mul_ope) { // z = a*b
      mpz_t u, v;
      mpz_inits(u, v, NULL);
      // zx = ax*bx - ay*by
      mpz_mul(u, ax, bx); mpz_mod(u, u, p);
      mpz_mul(v, ay, by); mpz_mod(v, v, p);
      mpz_sub(zx, u, v);
      if (mpz_cmp_ui(zx, 0U) < 0) { mpz_add(zx, zx, p); }
      // zy = ax*by + ay*bx
      mpz_mul(u, ax, by); mpz_mod(u, u, p);
      mpz_mul(v, ay, bx); mpz_mod(v, v, p);
      mpz_add(zy, u, v);
      if (mpz_cmp(zy, p) > 0) { mpz_sub(zy, zy, p); }
      mpz_clears(u, v, NULL);
    } else if (OPE == div_ope) { // z = a/b
      // Assume b.x != 0 and b.y != 0
      mpz_t ux, uy, vx, vy, inv_deno, t0, t1;
      mpz_inits(ux, uy, vx, vy, inv_deno, t0, t1, NULL);
      // u.x = b.x / (b.x^2 + b.y^2)
      mpz_mul(t0, bx, bx); mpz_mod(t0, t0, p);
      mpz_mul(t1, by, by); mpz_mod(t1, t1, p);
      mpz_add(inv_deno, t0, t1); mpz_mod(inv_deno, inv_deno, p);
      mpz_invert(inv_deno, inv_deno, p); 
      mpz_mul(ux, bx, inv_deno); mpz_mod(ux, ux, p); 
      // u.y = -b.y / (b.x^2 + b.y^2)      
      mpz_mul(uy, by, inv_deno); mpz_mod(uy, uy, p);       
      mpz_sub(uy, p, uy);      
      // z.x = a.x*u.x - a.y*u.y
      mpz_mul(t0, ax, ux); mpz_mod(t0, t0, p);
      mpz_mul(t1, ay, uy); mpz_mod(t1, t1, p);
      mpz_sub(zx, t0, t1);
      if (mpz_cmp_ui(zx, 0U) < 0) { mpz_add(zx, zx, p); }
      // z.y = a.x*u.y + a.y*u.x
      mpz_mul(t0, ax, uy); mpz_mod(t0, t0, p);
      mpz_mul(t1, ay, ux); mpz_mod(t1, t1, p);
      mpz_add(zy, t0, t1);
      if (mpz_cmp(zy, p) > 0) { mpz_sub(zy, zy, p); }
      mpz_clears(ux, uy, vx, vy, inv_deno, t0, t1, NULL);      
    } else if (OPE == madd_ope) { // z = a + b*c
      mpz_t t0, t1, wx, wy;
      mpz_inits(t0, t1, wx, wy, NULL);
      // z.x = a.x + (b.x*c.x - b.y*c.y)
      mpz_mul(t0, bx, cx); mpz_mod(t0, t0, p);
      mpz_mul(t1, by, cy); mpz_mod(t1, t1, p);
      mpz_sub(wx, t0, t1);
      if (mpz_cmp_ui(wx, 0U) < 0) { mpz_add(wx, wx, p); }
      mpz_add(zx, ax, wx);
      if (mpz_cmp(zx, p) > 0) { mpz_sub(zx, zx, p); }      
      // z.y = a.y + (b.x*c.y + b.y*c.x)
      mpz_mul(t0, bx, cy); mpz_mod(t0, t0, p);
      mpz_mul(t1, by, cx); mpz_mod(t1, t1, p);
      mpz_add(wy, t0, t1);
      if (mpz_cmp(wy, p) > 0) { mpz_sub(wy, wy, p); }
      mpz_add(zy, ay, wy);
      if (mpz_cmp(zy, p) > 0) { mpz_sub(zy, zy, p); }
      mpz_clears(t0, t1, wx, wy, NULL);
    } else {
      assert(false);
    }
    from_mpz(zx, correctx[i], NLIMBS);
    from_mpz(zy, correcty[i], NLIMBS);
  } // for (uint32_t i = 0; i < num_instances; ++i) {

  // Compare GMP results with GPU results.
  switch (OPE) {
    case  add_ope: std::cout << "Add  ";  break;
    case  sub_ope: std::cout << "Sub  ";  break;
    case  mul_ope: std::cout << "Mul  ";  break;
    case  div_ope: std::cout << "Div  ";  break;
    case madd_ope: std::cout << "MAdd  "; break;            
    default: assert(false);
  }
  if (OPE == add_ope)
  for (uint32_t i = 0; i < num_instances; ++i) {
    for (int32_t j = 0; j < NLIMBS; ++j) {
      if ((correctx[i][j] != instances[i].z.x._limbs[j])) {
        std::cout << "NG: x[" << i << "]["  << j << "]" << std::endl;
        std::cout << "GMP " << correctx[i][j] << std::endl;
        std::cout << "GPU " << instances[i].z.x._limbs[j] << std::endl;
        rc = false;
        goto clean_up;
      } else if ((correcty[i][j] != instances[i].z.y._limbs[j])) {
        std::cout << "NG: y[" << i << "]["  << j << "]" << std::endl;
        std::cout << "GMP " << correcty[i][j] << std::endl;
        std::cout << "GPU " << instances[i].z.y._limbs[j] << std::endl;
        rc = false;
        goto clean_up;
      }
    }
  }
  std::cout << "OK: all results" << std::endl;

clean_up:
  for (uint32_t i = 0; i < num_instances; ++i) {
    free(correctx[i]); free(correcty[i]);
  }
  free(correctx); free(correcty);
  mpz_clears(ax,ay, bx,by, cx,cy,  dx,dy,  ex,ey, fx,fy, zx,zy, p, NULL);
  
  return rc;
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
  typename cgbn_env_t<context_t, BITS>::cgbn2_t a, b, c, z;
  typename cgbn_env_t<context_t, BITS>::cgbn_t p;

  cgfp2_load(bn_env, a, &(instances[instance_idx].a), p);
  cgfp2_load(bn_env, b, &(instances[instance_idx].b), p);
  cgfp2_load(bn_env, c, &(instances[instance_idx].c), p);
  cgbn_load(bn_env, p, &(instances[instance_idx].p));
  if (OPE == add_ope) {
    cgfp2_add(bn_env, z, a, b, p);
  } else if (OPE == sub_ope) {
    cgfp2_sub(bn_env, z, a, b, p);
  } else if (OPE == mul_ope) {
    cgfp2_mul(bn_env, z, a, b, p);
  } else if (OPE == div_ope) {
    cgfp2_div(bn_env, z, a, b, p);
  } else if (OPE == madd_ope) {
    cgfp2_madd(bn_env, z, a, b, c, p);
    // cgfp2_madd<env_t>(bn_env, z, a, b, c, p); // OK
  } else {
    assert(false);
  }
  cgfp2_store(bn_env, &(instances[instance_idx].z), z, p);
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
  bool rc = verify_results(instances, NUM_INSTANCES);

  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpu_instances));
  CUDA_CHECK(cgbn_error_report_free(report));

  return (rc == true) ? 0 : 1;
} // main

// end of file
