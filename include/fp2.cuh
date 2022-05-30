//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//

/** @file
Quadratic extension field APIH
F_p^2 = F_p[x] / (x^2 + 1)
*/

#ifndef ISOGENY_REF_FP2_H
#define ISOGENY_REF_FP2_H

#include "cgbn_ext.cuh"
#include "fp.cuh"

/**
 * Data type for field-p2 elements: x0 + i*x1
 */
template<typename env_t>
struct fp2 {
  typename env_t::cgbn_t x0;
  typename env_t::cgbn_t x1;
};


template<typename env_t>
__host__ __device__  void
fp2_Clear(env_t env,
    const typename env_t::cgbn_t& p,
    cgbn2_t<env_t>& a){
        return;
    }


template<typename env_t>
__host__ __device__  void
fp2_Init(env_t env,
    const typename env_t::cgbn_t& p,
    cgbn2_t<env_t>& a){
        return;
    }

template<typename env_t>
__host__ __device__  void
fp2_Init_Set(env_t env,
    const typename env_t::cgbn_t& p,
    cgbn2_t<env_t>& a,
    unsigned long x0,
    unsigned long x1){
        cgfp2_set_ui32(env, a, x0, x1, p);
    }

template<typename env_t>
__host__ __device__  void
fp2_Init_Set_Ui32(env_t env,
    const typename env_t::cgbn_t& p,
    cgbn2_t<env_t>& a,
    uint32_t x0,
    uint32_t x1){
        cgfp2_set_ui32(env, a, x0, x1, p);
    }

template<typename env_t>
__host__ __device__  bool
fp2_IsEqual(env_t env,
    const typename env_t::cgbn_t& p,
    const cgbn2_t<env_t>& a,
    const cgbn2_t<env_t>& b){
        return cgfp2_equals(env, a, b, p);
    }

template<typename env_t>
__host__ __device__ void
fp2_Set(env_t env,
    const typename env_t::cgbn_t& p,
    const cgbn2_t<env_t>& a,
    cgbn2_t<env_t>& b){
        cgfp2_set(env, b, a, p);
    }


// Why the last argment is not an updated argument?
template<typename env_t>
__host__ __device__  void
fp2_Set(env_t env,
    const typename env_t::cgbn_t& p,
    cgbn2_t<env_t>& a,
    const uint32_t x0,
    const uint32_t x1){
        cgfp2_set_ui32(env, a, x0, x1, p);
    }

template<typename env_t>
__host__ __device__ void
fp2_Set_Ui32(env_t env,
    const typename env_t::cgbn_t& p,
    const uint32_t x0,
    const uint32_t x1,
    cgbn2_t<env_t>& a){
        cgfp2_set_ui32(env, a, x0, x1, p);
    }

/**
 * Subtraction in fp2
 * c = a-b
 *
 * @param p Finite field parameters
 * @param a Minuend
 * @param b Subtrahend
 * @param c Difference
 */
template<typename env_t>
__host__ __device__ bool
fp2_Add(env_t env,
    const typename env_t::cgbn_t& p,
    const cgbn2_t<env_t>& a,
    const cgbn2_t<env_t>& b,
    typename cgbn2_t<env_t>& c){
        return cgfp2_add(env, c, a, b, p);
    }


/**
 * Subtraction in fp2
 * c = a-b
 *
 * @param p Finite field parameters
 * @param a Minuend
 * @param b Subtrahend
 * @param c Difference
 */
template<typename env_t>
__host__ __device__ void
fp2_Sub(env_t env,
    const typename env_t::cgbn_t& p,
    const cgbn2_t<env_t>& a,
    const cgbn2_t<env_t>& b,
    cgbn2_t<env_t>& c){
        cgfp2_sub(env, c, a, b, p);
    }

/**
 * Multiplication in fp2
 * c = a*b
 *
 * @param p Finite field parameters
 * @param a First factor
 * @param b Second factor
 * @param c Product
 */
template<typename env_t>
__host__ __device__  void
fp2_Multiply(env_t env,
    const typename env_t::cgbn_t& p,
    const cgbn2_t<env_t>& a,
    const cgbn2_t<env_t>& b,
    cgbn2_t<env_t>& c){
        cgfp2_mul(env, c, a, b, p);
    }

/**
 * Squaring in fp2
 * b = a^2
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 */
template<typename env_t>
__host__ __device__  void
fp2_Square(env_t env,
    const typename env_t::cgbn_t& p,
    const cgbn2_t<env_t>& a,
    cgbn2_t<env_t>& b){
        cgfp2_sqr(env, b, a, p);
    }

/**
 * Inversion in fp2
 * b = a^-1
 *
 * @param p Finite field parameters
 * @param a Fp2 element to be inverted
 * @param b Inverted fp2 element
 */
template<typename env_t>
__host__ __device__ void
fp2_Invert(env_t env,
    const typename env_t::cgbn_t& p,
    const cgbn2_t<env_t>& a,
    cgbn2_t<env_t>& b){
        cgfp2_modular_inverse(env, b, a, p);
    }

/**
 * Negation in fp2
 * b = -a
 *
 * @param p Finite field parameters
 * @param a Fp2 element to be negated
 * @param b Negated fp2 element
 */
template<typename env_t>
__host__ __device__  void
fp2_Negative(env_t env,
    const typename env_t::cgbn_t& p,
    const cgbn2_t<env_t>& a,
    cgbn2_t<env_t>& b){
        cgfp2_negate(env, b, a, p);
    }

/**
 * Copying one fp2 element to another.
 * b = a
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 */
template<typename env_t>
__host__ __device__  void
fp2_Copy(env_t env,
    const typename env_t::cgbn_t& p,
    const cgbn2_t<env_t>& a,
    cgbn2_t<env_t>& b){
        cgfp2_set(env, b, a, p);
    }


template<typename env_t>
__host__  __device__ void
fp2_Rand(env_t env,
    const typename env_t::cgbn_t& p,
    cgbn2_t<env_t>& a){

    }

/**
 * Checks if an fp2 element equals integer constants
 * x0 + i*x1 == a.x0 + i*a.x1
 *
 * @param p Finite field parameters
 * @param a Fp2 element
 * @param x0
 * @param x1
 * @return 1 if equal, 0 if not
 */
// The function name may not be good?
template<typename env_t>
__host__ __device__  bool
fp2_IsConst(env_t env,
    const typename env_t::cgbn_t& p,
    const cgbn2_t<env_t>& a,
    unsigned long x0,
    unsigned long x1){
        return cgfp2_equals_ui32(env, a, x0, x1, p);
    }


template<typename env_t>
__host__ __device__  bool
fp2_IsEqual(env_t env,
    const typename env_t::cgbn_t& p,
    const cgbn2_t<env_t>& a,
    const uint32_t x0,
    const uint32_t x1){
        return cgfp2_equals_ui32(env, a, x0, x1, p);
    }
/**
 * Square root in fp2.
 * b = sqrt(a).
 * Only supports primes that satisfy p % 4 == 1
 *
 * @param p Finite field parameters
 * @param a
 * @param b
 * @param sol 0/1 depending on the solution to be chosen
 * @return
 */
template<typename env_t>
__host__ __device__  bool
fp2_Sqrt(env_t env,
    const typename env_t::cgbn_t& p,
    const cgbn2_t<env_t>& a,
    cgbn2_t<env_t>& b,
    const int32_t sol = 0){
         cgbn2_t<env_t> s0, s1;
        bool rc = cgfp2_sqrt(env, s0, s1, a, p);
        assert(rc == true);
        if (rc == false) {
            // a does not have square roots.
            return false;
        }
        if (sol == 0) {
            cgfp2_set(env, b, s0, p);
        }
        else {
            cgfp2_set(env, b, s1, p);
        }
        return true;
    }

#endif /* ISOGENY_REF_FP2_H */
