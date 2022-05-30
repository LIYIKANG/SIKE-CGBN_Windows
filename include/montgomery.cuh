//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//

/** @file
Data structures and arithmetic for supersingular Montgomery curves
*/
#ifndef ISOGENY_REF_MONTGOMERY_H
#define ISOGENY_REF_MONTGOMERY_H

// ================================================
// C system headers (headers in angle brackets with the .h extension), e.g., <stdlib.h>

// C++ standard library headers (without file extension), e.g., <cstddef>
// This project's .h files
#include "fp2.cuh"


/////////////
// Data types
/////////////

/**
 * Represents a point on a (Montgomery) curve with x and y (montgomery.h)
 */
template<typename env_t>
struct mont_pt_t {
  // cgbn2_t (cgbn_cuda.h)
  cgbn2_t<env_t> x;
  cgbn2_t<env_t> y;
};


/**
 * Internal representation of a Montgomery curve with. (montgomery.h)
 * - Underlying finite field parameters
 * - Generators P and Q
 * - Coefficients a and b
 */
template<typename env_t>
struct mont_curve_int_t {
  // ff_Params (fp.h)
  ff_Params<env_t> ffData;
  // cgbn2_t (cgbn_cuda.h)
  cgbn2_t<env_t> a;
  cgbn2_t<env_t> b;
  // mont_pt_t (montgomery.h)
  mont_pt_t<env_t> P;
  mont_pt_t<env_t> Q;
  mont_pt_t<env_t> R;
};


/**
 * External representation of a Montgomery curve for the public keys. (montgomery.h)
 * - Underlying finite field parameters
 * - Projective x-coordinates of generators `P` and `Q`
 * - Projective x-coordinate of `R`=`P`-`Q`
 */
template<typename env_t>
struct sike_public_key_t {
  // ff_Params (fp.h)  
  ff_Params<env_t> ffData;
  // cgbn2_t (cgbn_cuda.h)
  cgbn2_t<env_t> xP;
  cgbn2_t<env_t> xQ;
  cgbn2_t<env_t> xR;
};


/////////////////////////////////////////////////////////////////////
// Initialization and deinitialization routines for Montgomery curves
/////////////////////////////////////////////////////////////////////
/**
 * Initialization of a point
 *
 * @param p Finite field parameters
 * @param pt Point to be initialized
 */
template<typename env_t>
__host__ __device__ void
mont_pt_init(env_t env,
    const typename env_t::cgbn_t& p,
    mont_pt_t<env_t>& pt) {
  fp2_Init<env_t>(env, p, pt.x);
  fp2_Init<env_t>(env, p, pt.y);
}

template<typename env_t>
__host__ __device__ void
 mont_pt_init_set_ui32(env_t env,
    const typename env_t::cgbn_t& p,
    const uint32_t x_x0,
    const uint32_t x_x1,
    const uint32_t y_x0,
    const uint32_t y_x1,
    mont_pt_t<env_t>& pt) {
  fp2_Init_Set_Ui32<env_t>(env, pt.x, x_x0, x_x1);
  fp2_Init_Set_Ui32<env_t>(env, pt.y, y_x0, y_x1);
}


/**
 * Deinitialization of a point
 *
 * @param p Finite field parameters
 * @param pt Point to be deinitialized
 */
template<typename env_t>
__host__ __device__ void
mont_pt_clear(env_t env,
    const typename env_t::cgbn_t& p,
    mont_pt_t<env_t>& pt) {
  fp2_Clear<env_t>(env, p, pt.x);
  fp2_Clear<env_t>(env, p, pt.y);
}

/**
 * Copies a point. dst := src
 *
 * @param p Finite field parameters
 * @param src Source point
 * @param dst Destination point
 */
template<typename env_t>
__host__ __device__ void
mont_pt_copy(env_t env,
    const typename env_t::cgbn_t& p,
    const mont_pt_t<env_t>& src,
    mont_pt_t<env_t>& dst) {
  if (src != dst) {
    fp2_Copy<env_t>(env, p, src.x, dst.x);
    fp2_Copy<env_t>(env, p, src.y, dst.y);
  }
  else {
    // do nothing
  }
}

/**
 * Initialization of a curve
 *
 * @param p Finite field parameters
 * @param curve Curve to be initialized
 */
template<typename env_t>
__host__ __device__ void
mont_curve_init(env_t env,
    ff_Params<env_t>& p,
    mont_curve_int_t<env_t>& curve) {
  curve.ffData = p;
  mont_pt_init<env_t>(env, p, curve.P);
  mont_pt_init<env_t>(env, p, curve.Q);
  fp2_Init<env_t>(env, p.mod, curve.a);
  fp2_Init<env_t>(env, p.mod, curve.b);
}

/**
 * Deinitialization of a curve
 *
 * @param p Finite field parameters
 * @param curve Curve to be deinitialized
 */
template<typename env_t>
__host__ __device__ void
mont_curve_clear(env_t env,
    const typename env_t::cgbn_t& p,
    mont_curve_int_t<env_t>& curve) {
  mont_pt_clear<env_t>(env, p, curve.P);
  mont_pt_clear<env_t>(env, p, curve.Q);
  fp2_Clear<env_t>(env, p.mod, curve.a);
  fp2_Clear<env_t>(env, p.mod, curve.b);
}

/**
 * Copies a curve, curvecopy := curve
 * @param p Finite field parameters
 * @param curve Source curve
 * @param curvecopy Destination curve
 */
template<typename env_t>
__host__ __device__ void
mont_curve_copy(env_t env,
    const typename env_t::cgbn_t& p,
    const mont_curve_int_t<env_t>& src,
    mont_curve_int_t<env_t>& dst) {
  if (src != dst) {
    mont_pt_copy<env_t>(env, p, src.P, dst.P);
    mont_pt_copy<env_t>(env, p, src.Q, dst.Q);
    fp2_Copy<env_t>(env, p, src.a, dst.a);
    fp2_Copy<env_t>(env, p, src.b, dst.b);
  }
  else {
    // do nothing
  }
}

/**
 * Initialization of a public key
 *
 * @param p Finite field parameters (fp.h)
 * @param pk Public key to be initialized (montgomery.h)
 */
template<typename env_t>
__host__ __device__ void
public_key_init(env_t env,
    ff_Params<env_t>& p,
    sike_public_key_t<env_t>& pk) {
  pk.ffData = p;
  fp2_Init<env_t>(env, p, pk.xP);
  fp2_Init<env_t>(env, p, pk.xQ);
  fp2_Init<env_t>(env, p, pk.xR);
}


/**
 * Deinitialization of a public key
 *
 * @param p Finite field parameters (fp.h)
 * @param pk Public key to be deinitialized (montgomery.h)
 */
template<typename env_t>
__host__ __device__ void
public_key_clear(env_t env,
    const typename env_t::cgbn_t& p,
    sike_public_key_t<env_t>& pk) {
  fp2_Clear(p, pk.xP);
  fp2_Clear(p, pk.xQ);
  fp2_Clear(p, pk.xR);
}


/* infinity is represented as a point with (0, 0) */
template<typename env_t>
__host__ __device__ void
mont_set_inf_affine(env_t env,
    const mont_curve_int_t<env_t>& curve,
    mont_pt_t<env_t>& P) {
  const ff_Params<env_t> p = curve.ffData;
  fp2_Set_Ui32<env_t>(env, p, P.x, 0U, 0U);
  fp2_Set_Ui32<env_t>(env, p, P.y, 0U, 0U);
}


template<typename env_t>
__host__ __device__  bool
mont_is_inf_affine(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P) {
  return fp2_IsConst<env_t>(env, curve.ffData.mod, P.y, 0U, 0U);
}


///////////////////////////////////////////////
// Montgomery curve arithmetic - affine version
///////////////////////////////////////////////

/**
 * Scalar multiplication using the double-and-add.
 * Note: add side-channel countermeasures for production use.
 *
 * @param curve Underlying curve (mont_curve_int_t: montogomery.h)
 * @param k Scalar (cgbn_t: cgbn.h)
 * @param P Point (mont_pt_t: montgomery.h)
 * @param Q Result Q=kP (mont_pt_t: montgomery.h)
 * @param msb Most significant bit of scalar 'k' (int32_t)
 */
// forward declaration, the definition is given after 30 lines below.
template<typename env_t>
__host__ __device__ void
xDBL(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P,
    mont_pt_t<env_t>& R);


template<typename env_t>
__host__ __device__ void
mont_double_and_add(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const typename env_t::cgbn_t& k,
    const mont_pt_t<env_t>& P,
    mont_pt_t<env_t>& Q,
    const int32_t msb) {
  assert(msb >= 0);
  const ff_Params<env_t> p = curve.ffData;
  mont_pt_t<env_t> kP;
  fp2_Init_Set_Ui32(env, p.mod, kP.x, 0U, 0U);
  fp2_Init_Set_Ui32(env, p.mod, kP.y, 0U, 0U);
  mont_pt_init<env_t>(env, p, kP);
  mont_set_inf_affine<env_t>(env, curve, kP);
  for (int32_t i = msb - 1; i >= 0; --i) {
    xDBL<env_t>(env, curve, kP, kP);
    if (fp_IsBitSet<env_t>(env, p.mod, k, static_cast<uint32_t>(i))) {
      xADD(env, curve, kP, P, kP);
    }
  }
  mont_pt_copy<env_t>(env, p, kP, Q);
  mont_pt_clear<env_t>(env, p, kP);
}


/**
 * Affine doubling.
 *
 * @param curve Underlying curve
 * @param P Point
 * @param R Result R=2P
 */
template<typename env_t>
__host__ __device__ void
xDBL(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P,
    mont_pt_t<env_t>& R) {
  const typename env_t::cgbn_t p = curve.ffData.mod;
  const cgbn2_t<env_t> a = curve.a;
  const cgbn2_t<env_t> b = curve.b;

  if (mont_is_inf_affine<env_t>(env, curve, P)) {
    mont_set_inf_affine<env_t>(env, curve, R);
    return;
  }

  // Goal
  // x3 := b*(3*x1^2+2*a*x1+1)^2/(2*b*y1)^2-a-2*x1
  // y3 := (3*x1+a)*(3*x1^2+2*a*x1+1)/(2*b*y1)-b*(3*x1^2+2*a*x1+1)^3/(2*b*y1)^3-y1
  cgbn2_t<env_t> t0, t1, t2;
  fp2_Set_Ui32<env_t>(env, p, t0, 0U, 0U);
  fp2_Set_Ui32<env_t>(env, p, t1, 0U, 0U);
  fp2_Set_Ui32<env_t>(env, p, t2, 0U, 0U);
  // t2 = 1
  fp2_Set_Ui32<env_t>(env, p, &t2, 1U, 0U);
  // t0 = x1^2
  fp2_Square<env_t>(env, p, P.x, t0);
  // t1 := t0+t0 = (x1^2)+(x1^2) = 2*x1^2
  fp2_Add<env_t>(env, p, t0, t0, t1);
  // t0 := t0+t1 = (x1^2)+(2*x1^2) = 3*x1^2
  fp2_Add<env_t>(env, p, t0, t1, t0);
  // t1 := a*x1
  fp2_Multiply<env_t>(env, p, a, P.x, t1);
  // t1 := t1+t1 = (a*x1)+(a*x1) = 2*a*x1
  fp2_Add<env_t>(env, p, t1, t1, t1);
  // t0 := t0+t1 = (3*x1^2)+(2*a*x1),
  fp2_Add<env_t>(env, p, t0, t1, t0);
  // t0 := t0+t2 = 3*x1^2+2*a*x1 + 1 =
  fp2_Add<env_t>(env, p, t0, t2, t0);
  // t1 := b*y1
  fp2_Multiply<env_t>(env, p, b, P.y, t1);
  // t1 := t1+t1 = (b*y1)+(b*y1) = 2*b*y1
  fp2_Add<env_t>(env, p, t1, t1, t1);
  // t1 := 1/t1 = 1/(2*b*y1)
  fp2_Invert<env_t>(env, p, t1, t1);
  // t0 := t0*t1 = (3*x1^2+2*a*x1+1) / (2*b*y1)
  fp2_Multiply<env_t>(env, p, t0, t1, t0);
  // t1 := t0^2 = ((3*x1^2+2*a*x1+1)/(2*b*y1))^2
  fp2_Square<env_t>(env, p, t0, t1);
  // t2 := b*(3*x1^2+2*a*x1+1)^2/(2*b*y1)^2
  fp2_Multiply<env_t>(env, p, b, t1, t2);
  // t2 := b*(3*x1^2+2*a*x1+1)^2/(2*b*y1)^2 - a
  fp2_Sub<env_t>(env, p, t2, a, t2);
  // t2 := t1-x1 = b*(3*x1^2+2*a*x1+1)^2/(2*b*y1)^2-a - x1
  fp2_Sub<env_t>(env, p, t2, P.x, t2);
  // t2 := t2-x1 = b*(3*x1^2+2*a*x1+1)^2/(2*b*y1)^2-a-x1 - x1
  // = b*(3*x1^2+2*a*x1+1)^2/(2*b*y1)^2-a-2*x1
  fp2_Sub<env_t>(env, p, t2, P.x, t2);
  // t1 := t0*t1 = (3*x1^2+2*a*x1+1)/(2*b*y1) * (3*x1^2+2*a*x1+1)^2/(2*b*y1)^2
  // = (3*x1^2+2*a*x1+1)^3/(2*b*y1)^3
  fp2_Multiply<env_t>(env, p, t0, t1, t1);
  // t1: = b * (3*x1^2+2*a*x1+1)^3/(2*b*y1)^3
  fp2_Multiply<env_t>(env, p, b, t1, t1);
  // t1 := b*(3*x1^2+2*a*x1+1)^3/(2*b*y1)^3 + y1
  fp2_Add<env_t>(env, p, t1, P.y, t1);
  // y3 := x1 + x1 = 2*x1
  fp2_Add<env_t>(env, p, P.x, P.x, R.y);
  // y3 := y3 + x1 = 2*x1 + x1 = 3*x1
  fp2_Add<env_t>(env, p, R.y, P.x, R.y);
  // y3 := y3 + a = 3*x1 + a
  fp2_Add<env_t>(env, p, R.y, a, R.y);
  // y3 := y3 * t0 = (3*x1+a) * (3*x1^2+2*a*x1+1)/(2*b*y1)
  fp2_Multiply<env_t>(env, p, R.y, t0, R.y);
  // y3 := y3 - t1
  // = (3*x1+a)*(3*x1^2+2*a*x1+1)/(2*b*y1) - (b*(3*x1^2+2*a*x1+1)^3/(2*b*y1)^3+y1)
  fp2_Sub<env_t>(env, p, R.y, t1, R.y);
  // x3 := t2 = b*(3*x1^2+2*a*x1+1)^2/(2*b*y1)^2-a-2*x1
  fp2_Copy<env_t>(env, p, t2, R.x);

  fp2_Clear<env_t>(env, p, t0);
  fp2_Clear<env_t>(env, p, t1);
  fp2_Clear<env_t>(env, p, t2);
}


/**
 * Repeated affine doubling (montgomery.h)
 *
 * @param curve Underlying curve (mont_curve_int_t: montgomery.h)
 * @param P Point (mont_pt_t: montgomery.h)
 * @param e Repetitions (int32_t)
 * @param R Result R=2^e*P (mont_pt_t: montgomery.h)
 */
template<typename env_t>
__host__ __device__ void
xDBLe(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P,
    const int32_t e,
    mont_pt_t<env_t>& R) {
  assert(e >= 0);
  mont_pt_copy<env_t>(curve.ffData, P, R);
  for (int32_t j = 0; j < e; ++j) {
    xDBL<env_t>(env, curve, R, R);
  }
}


/**
 * Affine addition. (montgomey.h)
 *
 * @param curve Underlying curve (mont_curve_int_t: montgomery.h)
 * @param P First point (mont_pt_t: montgomery.h)
 * @param Q Second point (mont_pt_t: montgomery.h)
 * @param R Result R=P+Q (mont_pt_t: montgomery.h)
 */
template<typename env_t>
__host__ __device__ void
xADD(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P,
    const mont_pt_t<env_t>& Q,
    mont_pt_t<env_t>& R) {
  // x3 = b*(y2-y1)^2/(x2-x1)^2-a-x1-x2
  // y3 = (2*x1+x2+a)*(y2-y1)/(x2-x1)-b*(y2-y1)^3/(x2-x1)^3-y1
  // y3 = ((2*x1)+x2+a) * ((y2-y1)/(x2-x1)) - b*((y2-y1)^3/(x2-x1)^3) - y1

  const ff_Params<env_t> p = curve.ffData;
  const cgbn2_t<env_t> a = curve.a;
  const cgbn2_t<env_t> b = curve.b;

  cgbn2_t<env_t> t0, t1, t2;
  fp2_Set_Ui32<env_t>(env, p.mod, t0, 0U, 0U);
  fp2_Set_Ui32<env_t>(env, p.mod, t1, 0U, 0U);
  fp2_Set_Ui32<env_t>(env, p.mod, t2, 0U, 0U);

  fp2_Negative<env_t>(env, p.mod, Q.y, t0);

  if (mont_is_inf_affine<env_t>(env, curve, P)) {
    mont_pt_copy<env_t>(env, p, Q, R);
  }
  else if (mont_is_inf_affine<env_t>(env, curve, Q)) {
    mont_pt_copy<env_t>(env, p, P, R);
  }
  else if (fp2_IsEqual<env_t>(env, p, P.x, Q.x) && fp2_IsEqual<env_t>(env, p, P.y, Q.y)) {
    /* P == Q */
    xDBL<env_t>(env, curve, P, R);
  }
  else if (fp2_IsEqual<env_t>(env, p, P->x, Q->x) && fp2_IsEqual<env_t>(env, p, P->y, t0)) {
    /* P == -Q */
    mont_set_inf_affine<env_t>(env, curve, R);
  }
  else {
    /* P != Q and P != -Q  */
    // t0 := y2-y1
    fp2_Sub<env_t>(env, p, Q.y, P.y, t0);
    // t1 := x2-x1
    fp2_Sub<env_t>(env, p, Q.x, P.x, t1);
    // t1 := 1/t1 = 1/(x2-x1)
    fp2_Invert<env_t>(env, p, t1, t1);
    // t0 := t0 * t1 = (y2-y1) * 1/(x2-x1) = (y2-y1)/(x2-x1)
    fp2_Multiply<env_t>(env, p, t0, t1, t0);
    // t1 := t0^2 = (y2-y1)^2/(x2-x1)^2
    fp2_Square<env_t>(env, p, t0, t1);
    // t2 := x1 + x1 = 2*x1
    fp2_Add<env_t>(env, p, P.x, P.x, t2);
    // t2 := t2 + x2 = 2*x1 + x2
    fp2_Add<env_t>(env, p, t2, Q.x, t2);
    // t2 := t2 + a = 2*x1+x2 + a
    fp2_Add<env_t>(env, p, t2, a, t2);
    // t2 := t2 * t0 = (2*x1+x2+a) * (y2-y1)/(x2-x1)
    fp2_Multiply<env_t>(env, p, t2, t0, t2);
    // t0 := t0 * t1 = (y2-y1)/(x2-x1) * (y2-y1)^2/(x2-x1)^2 = (y2-y1)^3/(x2-x1)^3
    fp2_Multiply<env_t>(env, p, t0, t1, t0);
    // t0 := b * t0 = b * (y2-y1)^3/(x2-x1)^3
    fp2_Multiply<env_t>(env, p, b, t0, t0);
    // t0 := t0 + y1 = b*(y2-y1)^3/(x2-x1)^3 + y1
    fp2_Add<env_t>(env, p, t0, P.y, t0);
    // t0 := t2 - t0 = (2*x1+x2+a)*(y2-y1)/(x2-x1) - b*(y2-y1)^3/(x2-x1)^3-y1
    fp2_Sub<env_t>(env, p, t2, t0, t0);
    // t1 := b * t1 = b * (y2-y1)^2/(x2-x1)^2
    fp2_Multiply<env_t>(env, p, b, t1, t1);
    // t1 := t1 - a = b*(y2-y1)^2/(x2-x1)^2 - a
    fp2_Sub<env_t>(env, p, t1, a, t1);
    // t1 := t1 - x1 = b*(y2-y1)^2/(x2-x1)^2-a - x1
    fp2_Sub<env_t>(env, p, t1, P.x, t1);
    // x3 := t1 - x2 = b*(y2-y1)^2/(x2-x1)^2-a-x1 - x2
    fp2_Sub<env_t>(env, p, t1, Q.x, R.x);
    // y3 := t0 = (2*x1+x2+a)*(y2-y1)/(x2-x1)-(b*(y2-y1)^3/(x2-x1)^3+y1)
    fp2_Copy<env_t>(env, p, t0, R.y);
  }
  fp2_Clear<env_t>(env, p, t0);
  fp2_Clear<env_t>(env, p, t1);
  fp2_Clear<env_t>(env, p, t2);
}


/**
 * Affine tripling. (montgomery.h)
 *
 * @param curve Underlying curve (mont_curve_int_t: montgomery.h)
 * @param P Point (mont_pt_t: mongomery.h)
 * @param R Result R=3P (mont_pt_t: mongomery.h)
 */
template<typename env_t>
__host__ __device__ void
xTPL(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P,
    mont_pt_t<env_t>& R) {
  const ff_Params<env_t> p = curve.ffData;
  mont_pt_t<env_t> T;
  mont_pt_init_set_ui32(env, p, 0U, 0U, 0U, 0U, T);
  xDBL(env, curve, P, T);
  xADD(env, curve, P, T, R);
  mont_pt_clear(env, p, T);
}


/**
 * Repeated affine tripling (montgomery.h)
 *
 * @param curve Underlying curve (mont_curve_int_t: montgomery.h)
 * @param P Point (mont_pt_t: mongomery.h)
 * @param e Repetitions (int32_t)
 * @param R Result R=3^e*P (mont_pt_t: mongomery.h)
 */
template<typename env_t>
__host__ __device__ void
xTPLe(env_t env,
    const mont_curve_int_t<env_t>& curve,
    const mont_pt_t<env_t>& P,
    const int32_t e,
    mont_pt_t<env_t>& R) {
  assert(e >= 0);
  mont_pt_copy<env_t>(env, curve.ffData, P, R);
  for (int32_t j = 0; j < e; ++j) {
    xTPL<env_t>(env, curve, R, R);
  }
}


/**
 * J-invariant of a montgomery curve (montgomery.h)
 *
 * jinv = 256*(a^2-3)^3/(a^2-4);
 *
 * @param p Finite field parameters (ff_Params: fp.h)
 * @param E Montgomery curve (mont_curve_int_t: montgomery.h)
 * @param jinv Result: j-invariant (cgbn2_t: fp2.h)
 */
template<typename env_t>
__host__ __device__ void
j_inv(env_t env,
    const typename env_t::cgbn_t& p,
    const mont_curve_int_t<env_t>& E,
    cgbn2_t<env_t>& jinv) {
  const cgbn2_t<env_t> a = E.a;
  cgbn2_t<env_t> t0, t1;
  fp2_Init_Set_Ui32(env, p.mod, t0, 0U, 0U);
  fp2_Init_Set_Ui32(env, p.mod, t1, 0U, 0U);
  // t0 := a^2
  fp2_Square<env_t>(env, p.mod, a, t0);
  // jinv := 3
  fp2_Set_Ui32<env_t>(env, p.mod, jinv, 3U, 0U);
  // jinv := t0 - jinv = a^2 - 3
  fp2_Sub<env_t>(env, p.mod, t0, jinv, jinv);
  // t1 := jinv^2 = (a^2-3)^2
  fp2_Square<env_t>(env, p.mod, jinv, t1);
  // jinv := jinv * t1 = (a^2-3) * (a^2-3)^2 = (a^2-3)^3
  fp2_Multiply<env_t>(env, p.mod, jinv, &t1, jinv);
  // jinv := jinv + jinv = 2*(a^2-3)^3
  fp2_Add<env_t>(env, p.mod, jinv, jinv, jinv);
  // jinv := jinv + jinv = 4*(a^2-3)^3
  fp2_Add<env_t>(env, p.mod, jinv, jinv, jinv);
  // jinv := 8*(a^2-3)^3
  fp2_Add<env_t>(env, p.mod, jinv, jinv, jinv);
  // jinv := 16*(a^2-3)^3
  fp2_Add<env_t>(env, p.mod, jinv, jinv, jinv);
  // jinv := 32*(a^2-3)^3
  fp2_Add<env_t>(env, p.mod, jinv, jinv, jinv);
  // jinv := 64*(a^2-3)^3
  fp2_Add<env_t>(env, p.mod, jinv, jinv, jinv);
  // jinv := 128*(a^2-3)^3
  fp2_Add<env_t>(env, p.mod, jinv, jinv, jinv);
  // jinv := 256*(a^2-3)^3
  fp2_Add<env_t>(env, p.mod, jinv, jinv, jinv);
  // t1 := 4
  fp2_Set_Ui32<env_t>(env, p.mod, t1, 4U, 0U);
  // t0 := t0 - t1 = a^2 - 4
  fp2_Sub<env_t>(env, p.mod, t0, t1, t0);
  // t0 := 1/t0 = 1/(a^2-4)
  fp2_Invert<env_t>(env, p.mod, t0, t0);
  // jinv := jinv * t0 = 256*(a^2-3)^3 / (a^2-4)
  fp2_Multiply<env_t>(env, p.mod, jinv, t0, jinv);

  fp2_Clear<env_t>(env, p.mod, t0);
  fp2_Clear<env_t>(env, p.mod, t1);
}


/**
 * Conversion of a Montgomery curve with affine parameters to the external format for public keys (montgomery.h)
 *
 * a, b, P.x, P.y, Q.x, Q.y -> P.x, Q.x, (P-Q).x
 *
 * @param p Finite field arithmetic (ff_Params: fp.h)
 * @param curve Montgomery curve (mont_curve_int_t: montgomery.h)
 * @param pk Public key parameters (sike_public_key_t: montgomery.h)
 */
template<typename env_t>
__host__ __device__ void 
get_xR(env_t env,
    const typename env_t::cgbn_t& p,
    const mont_curve_int_t<env_t>& curve,
    sike_public_key_t<env_t>& pk) {
  mont_pt_t<env_t> R;
  mont_pt_init_set_ui32<env_t>(env, p, 0U, 0U, 0U, 0U, R);

  mont_pt_copy<env_t>(env, p, curve.Q, R);
  fp2_Negative<env_t>(env, p.mod, R.y, R.y);

  xADD<env_t>(env, curve, curve.P, R, R);
  fp2_Copy<env_t>(env, p.mod, curve.P.x, pk.xP);
  fp2_Copy<env_t>(env, p, curve.Q.x, pk.xQ);
  fp2_Copy<env_t>(env, p, R.x, pk.xR);

  mont_pt_clear<env_t>(env, p, R);
}


/**
 * Conversion of public key parameters to the internal affine Montgomery curve parameters (montgomery.h)
 *
 * P.x, Q.x. (P-Q).x -> P.x, P.y, Q.x, Q.y, a, b
 *
 * @param p (ff_Params: fp.h)
 * @param pk (sike_public_key_t: montgomery.h)
 * @param curve (mont_curve_int_t: montgomery.h)
 */
template<typename env_t>
__host__ __device__ void
get_yP_yQ_A_B(env_t env,
    const typename env_t::cgbn_t& p,
    const sike_public_key_t<env_t> pk,
    mont_curve_int_t<env_t>& curve) {
  cgbn2_t<env_t> a = curve.a;
  cgbn2_t<env_t> b = curve.b;
  mont_pt_t<env_t> P = curve.P;
  mont_pt_t<env_t> Q = curve.Q;

  const cgbn2_t<env_t> xP = pk.xP;
  const cgbn2_t<env_t> xQ = pk.xQ;
  const cgbn2_t<env_t> xR = pk.xR;

  mont_pt_t<env_t> T;
  mont_pt_init_set_ui32<env_t>(env, p, 0U, 0U, 0U, 0U, T);
  cgbn2_t<env_t> t1 = T.x;
  cgbn2_t<env_t> t2 = T.y;

  // Goal a:=(1-xP*xQ-xP*xR-xQ*xR)^2/(4*xP*xQ*xR)-xP-xQ-xR;

  // a := xP * xQ
  fp2_Multiply<env_t>(env, p, xP, xQ, a);
  // t1 := a * xR = xP*xQ * xR
  fp2_Multiply<env_t>(env, p, a, xR, t1);
  // t1 := t1 + t1 = 2*xP*xQ*xR
  fp2_Add<env_t>(env, p, t1, t1, t1);
  //t1 := t1 + t1 = 4*xP*xQ*xR
  fp2_Add<env_t>(env, p, t1, t1, t1);
  // t1 := 1/t1 := 1/(4*xP*xQ*xR)
  fp2_Invert<env_t>(env, p, t1, t1);
  // t2 := 1
  fp2_Set_Ui32<env_t>(env, p, 1U, 0U, t2);
  // a := t2 - a = 1 - xP*xQ
  fp2_Sub<env_t>(env, p, t2, a, a);
  // t2 := xP * xR
  fp2_Multiply<env_t>(env, p, xP, xR, t2);
  // a := a - t2 = 1-xP*xQ - xP*xR
  fp2_Sub<env_t>(env, p, a, t2, a);
  // t2 := xQ * xR
  fp2_Multiply<env_t>(env, p, xQ, xR, t2);
  // a := a - t2 = 1-xP*xQ-xP*xR - xQ*xR
  fp2_Sub<env_t>(env, p, a, t2, a);
  // a := a^2 = (1-xP*xQ-xP*xR-xQ*xR)^2
  fp2_Square<env_t>(env, p, a, a);
  // a := a * t1 = (1-xP*xQ-xP*xR-xQ*xR)^2 / (4*xP*xQ*xR)
  fp2_Multiply<env_t>(env, p, a, t1, a);
  // a := a - xP = (1-xP*xQ-xP*xR-xQ*xR)^2/(4*xP*xQ*xR) - xP
  fp2_Sub<env_t>(env, p, a, xP, a);
  // a := a - xQ = (1-xP*xQ-xP*xR-xQ*xR)^2/(4*xP*xQ*xR)-xP - xQ
  fp2_Sub<env_t>(env, p, a, xQ, a);
  // a := a - xR = (1-xP*xQ-xP*xR-xQ*xR)^2/(4*xP*xQ*xR)-xP-xQ - xR
  fp2_Sub<env_t>(env, p, a, xR, a);
  // t1 := xP^2
  fp2_Square<env_t>(env, p, xP, t1);
  // t2 := xP * t1 = xP * xP^2 = xP^3
  fp2_Multiply<env_t>(env, p, xP, t1, t2);
  // t1 := a * t1 = a * xP^2
  fp2_Multiply<env_t>(env, p, a, t1, t1);
  // t1 := t2 + t1 = xP^3 + a*xP^2
  fp2_Add<env_t>(env, p, t2, t1, t1);
  // t1 := t1 + xP = xP^3+a*xP^2 + xP
  fp2_Add<env_t>(env, p, t1, xP, t1);
  // yP := sqrt(t1) = sqrt(xP^3+a*xP^2+xP) smaller?
  fp2_Sqrt<env_t>(env, p, t1, P.y, 0);
  // t1 := xQ^2
  fp2_Square<env_t>(env, p, xQ, t1);
  // t2 := xQ * t1 = xQ * xQ^2 = xQ^3
  fp2_Multiply<env_t>(env, p, xQ, t1, t2);
  // t1 := a * t1 = a * xQ^2
  fp2_Multiply<env_t>(env, p, a, t1, t1);
  // t1 := t2 + t1 = xQ^3 + a*xQ^2
  fp2_Add<env_t>(env, p, t2, t1, t1);
  // t1 := t1 + xQ = xQ^3+a*xQ^2 + xQ
  fp2_Add<env_t>(env, p, t1, xQ, t1);
  // yQ := sqrt(t1) = sqrt(xQ^3+a*xQ^2+xQ) smaller?
  fp2_Sqrt<env_t>(env, p, t1, Q.y, 0);
  // P.x := xP
  fp2_Copy<env_t>(env, p, xP, P.x);
  // Q.x := xQ
  fp2_Copy<env_t>(env, p, xQ, Q.x);
  // b := 1+0*i
  fp2_Set_Ui32<env_t>(env, p, b, 1, 0);
  // T := Q
  mont_pt_copy<env_t>(env, p, Q, T);
  // T.y := -T.y
  fp2_Negative<env_t>(env, p, T.y, T.y);
  // T:= P + T
  xADD<env_t>(env, curve, P, &T, &T);

  if (!fp2_IsEqual<env_t>(env, p, T.x, xR)) {
    fp2_Negative<env_t>(env, p, Q.y, Q.y);
  }

  mont_pt_clear<env_t>(env, p, T);
}

#endif //ISOGENY_REF_MONTGOMERY_H
