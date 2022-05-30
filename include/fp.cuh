#ifndef ISOGENY_REF_FP_H
#define ISOGENY_REF_FP_H

#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>

#include "assume.h"
#include "convert.h"

#include "gmp.h"
#include "cgbn/cgbn.h"

#include "cgbn_ext.cuh"
#include "encoding.cuh"
#include "rng.h"

/**
 * Type for multi-precision arithmetic
 */
typedef mpz_t mp;


/**
 * Finite field parameters and arithmetic, given the modulus.
 */
template<typename env_t>
struct ff_Params {
  /* The modulus */
  typename env_t::cgbn_t mod;
};

/**
 * Initializes the Finite field parameters with GMP implementations.
 * @param params Finite field parameters to be initialized.
 */
template<typename env_t>
void set_gmp_fp_params(ff_Params<env_t>& params){    
}

/**
 * Addition
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 * @param c =a+b (mod p)
 * void
fp_Add(const ff_Params *p, const mp a, const mp b, mp c);
 */
template<typename env_t>
__host__ __device__ int32_t
fp_Add(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    typename env_t::cgbn_t& c){
        return cgfp_add<env_t>(env, c, a, b, p);
}

/**
 * Clearing/deinitialization of an fp element
 *
 * @param p Finite field parameters
 * @param a To be cleared
 * void
fp_Clear(const ff_Params *p, mp a);
 */
template<typename env_t>
__host__ __device__ void
fp_Clear(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& a){
        return;
}

/**
 * Set to an integer constant
 *
 * @param p Finite field parameters
 * @param a integer constant
 * @param b MP element to be set
 * void
fp_Constant(const ff_Params *p, unsigned long a, mp b);
 */
template<typename env_t>
__host__ __device__ void
fp_Constant(env_t env,
    const typename env_t::cgbn_t& p,
    const unsigned long a,
    typename env_t::cgbn_t& b){
    cgfp_set_ui32(env, b, a, p); 
}

    /**
 * Copy one fp element to another.
 * dst = src
 *
 * @param p Finite field parameters
 * @param dst Destination
 * @param src Source
 * void
fp_Copy(const ff_Params *p, mp dst, const mp src);
 */
template<typename env_t>
__host__ __device__ void
fp_Copy(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& dst,
    const typename env_t::cgbn_t&src){
    cgfp_set(env, dst, src, p);
}

    /**
 * Initialization
 *
 * @param p Finite field parameters
 * @param a Element to be intiialized
 * void
fp_Init(const ff_Params* p, mp a);
 */

template<typename env_t>
__host__ __device__ void
fp_Init(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& a){
    return;
  }

  /**
 * Checking for equality
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 * @return 1 if a equals b, 0 otherwise
 * int
fp_IsEqual(const ff_Params *p, const mp a, const mp b);
 */  
template<typename env_t>
__host__ __device__ bool
fp_IsEqual(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b){
    return cgfp_equals(env, a, b, p);
  }

    /**
 * Inversion
 *
 * @param p Finite field parameters
 * @param a
 * @param b =a^-1 (mod p)
 * void
fp_Invert(const ff_Params *p, const mp a, mp b);

 */
template<typename env_t>
__host__ __device__ bool
fp_Invert(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    typename env_t::cgbn_t& b){
        return cgfp_modular_inverse(env, b, a, p);
    }

    /**
 * Checks if the i'th bit is set
 *
 * @param p Finite field parameters
 * @param a
 * @param i index
 * @return 1 if i'th bit in a is set, 0 otherwise
 * int
fp_IsBitSet(const ff_Params *p, const mp a, const unsigned long i);
 */
template<typename env_t>
__host__ __device__ bool
fp_IsBitSet(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const unsigned long index){
        uint32_t bit = cgbn_extract_bits_ui32(env, a, index, 1);
        return (bit == 0x00000001U) ? true : false;
    }

    /**
 * Checks equality with an integer constant
 *
 * @param p Finite field parameters
 * @param a
 * @param constant
 * @return 1 if a == constant, 0 otherwise
 * int
fp_IsConstant(const ff_Params *p, const mp a, const size_t constant);

 */

template<typename env_t>
__host__ __device__ void
fp_IsConstant(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const size_t constant){
        cgfp_equals_ui32(env, a, constant, p); 
    }

    /**
 * Multiplication
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 * @param c =a*b (mod p)
 * void
fp_Multiply(const ff_Params *p, const mp a, const mp b, mp c);

 */

template<typename env_t>
__host__ __device__ void
fp_Multiply(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    typename env_t::cgbn_t& c){
        cgfp_mul(env, c, a, b, p);
    }


/**
 * Negation
 *
 * @param p Finite field parameters
 * @param a
 * @param b =-a (mod p)
 * void
fp_Negative(const ff_Params *p, const mp a, mp b);
 */

template<typename env_t>
__host__ __device__ void
fp_Negative(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    typename env_t::cgbn_t& b){
        cgfp_negate(env, b, a, p);
    }

/**
 * Exponentiation
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 * @param c = a^b (mod p)
 * void
fp_Pow(const ff_Params *p, const mp a, const mp b, mp c);
 */
template<typename env_t>
__host__ __device__ void
fp_Pow(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    typename env_t::cgbn_t& c){
        cgfp_modular_poer(env, c, a, b, p);
    }


__host__  inline void
convert_array_to_mpz(const uint32_t nlimbs,
    const uint32_t* limbs,
    mpz_t a) {
  mpz_import(a, nlimbs, -1, sizeof(uint32_t), 0, 0, limbs);
}

/**
 * Generation of a random element in {0, ..., p->modulus - 1}
 *
 * @param p Finite field parameters
 * @param a Random element in {0, ..., p->modulus - 1}
 * void
fp_Rand(const ff_Params *p, mp a);
 */

template<typename env_t>
__host__ __device__ void
fp_Rand(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& a,
    const bool generation_mode = true){
// cgbn_t -> mpz_t, because mpz_urandomm() is used.
  static bool isFirstCall = true;
  static gmp_randstate_t state;
  if (isFirstCall) {//第一次调用时需要生成
    // low-quality, insecure random
    const uint32_t seed = 220322U;
    gmp_randseed_ui(state, seed);
    gmp_randinit_default(state);//解放
    isFirstCall = false;
  }
  if (generation_mode == false) {
    if (isFirstCall) {
      return;
    }
    else {
      gmp_randclear(state);//解放
      isFirstCall = true;
      return;
    }
  }
  // random generator using GMP functions
  mpz_t p_mpz, a_mpz;
  mpz_inits(p_mpz, a_mpz, NULL);
  convert_array_to_mpz(env.LIMBS, p._limbs, p_mpz);
  mpz_urandomm(a_mpz, state, p_mpz);
  convert_mpz_to_array(a_mpz, env.LIMBS, a._limbs); 
  mpz_clears(p_mpz, a_mpz, NULL);
}


    /**
 * Squaring
 *
 * @param p Finite field parameters
 * @param a
 * @param b =a^2 (mod p)
 * void
fp_Square(const ff_Params *p, const mp a, mp b);

 */

template<typename env_t>
__host__ __device__ void
fp_Square(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    typename env_t::cgbn_t& b){
        cgfp_sqr(env, b, a, p);
    }


/**
 * Subtraction
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 * @param c =a-b (mod p)
 * void
fp_Subtract(const ff_Params *p, const mp a, const mp b, mp c);
 */
template<typename env_t>
__host__ __device__ void
fp_Subtract(env_t env,
    const typename env_t::cgbn_t& p,
    const typename env_t::cgbn_t& a,
    const typename env_t::cgbn_t& b,
    typename env_t::cgbn_t& c){
        cgfp_sub(env, c, a, b, p);
    }


    /**
 * Set to unity (1)
 *
 * @param p Finite field parameters
 * @param b = 1
 * void
fp_Unity(const ff_Params *p, mp b);

 */

template<typename env_t>
__host__ __device__ void
fp_Unity(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& a){
        cgfp_set_ui32(env, a, 1U, p);
    }


/**
 * Set to zero
 *
 * @param p Finite field parameters
 * @param a = 0
 * void
fp_Zero(const ff_Params *p, mp a);

 */
template<typename env_t>
__host__ __device__ void
fp_Zero(env_t env,
    const typename env_t::cgbn_t& p,
    typename env_t::cgbn_t& a){
        cgfp_set_ui32(env, a, 0U, p); 
    }

    /**
 * Decodes and sets an element to an hex value
 *
 * @param hexStr
 * @param a = hexString (decoded)
 * 
 * void
fp_ImportHex(const char *hexStr, mp a);
 */

template<typename env_t>
__host__ __device__ void
fp_ImportHex(env_t env,
    const char* hexStr,
    typename env_t::cgbn_t a){
        mpz_set_str(a, hexStr, 0);
    }

#endif //ISOGENY_REF_FP_H