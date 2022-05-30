//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//

/** @file
SIDH key agreement high-level functions:
 Private key generation
 Public key generation
 Shared secret generation
*/
#ifndef ISOGENY_REF_SIDH_H
#define ISOGENY_REF_SIDH_H

// C system headers (headers in angle brackets with the .h extension), e.g., <stdlib.h>
// C++ standard library headers (without file extension), e.g., <cstddef>
// Other libraries' .h files
// This project's .h files

#include "fp2.cuh"
#include "sike_params.cuh"
#include "sike.cuh"


/**
 * SIDH private-key generation (sidh.h)
 *
 * Private keys are integers in the range:
 * {0,...,ordA - 1} for ALICE
 * {0,...,ordB - 1} for BOB
 *
 * @param params SIKE parameters (sike_params_t: sike_params.h)
 * @param party ALICE or BOB (Party: sike.h)
 * @param sk Private key to be generated (sike_private_key_t: sike.h)
 */
template<typename env_t>
__host__ __device__ void 
sidh_sk_keygen(env_t env,
    const sike_params_t<env_t>& params,
    const party_t party,
    typename env_t::sike_private_key_t& sk) {
  return;
#if 0        
  size_t bytes = 0;
  unsigned char* arr = 0;

  if (party == ALICE) {
    // Outputs random value in [0, 2^lA - 1]
    size_t orderLen = mp_sizeinbase(params->ordA, 2);
    bytes = BITS_TO_BYTES_CEIL(orderLen);
    arr = (unsigned char*)malloc(bytes);
    randombytes(arr, bytes);
    ostoi(arr, bytes, sk);
    mpz_mod(sk, sk, params->ordA);
  } else {
    // Outputs random value in [0, 2^Floor(Log(2,3^lB)) - 1]
    size_t orderLen = mp_sizeinbase(params->ordB, 2);
    bytes = BITS_TO_BYTES_CEIL(orderLen - 1);
    arr = (unsigned char*)malloc(bytes);
    randombytes(arr, bytes);
    ostoi(arr, bytes, sk);
    mp mod;
    mpz_init_set_ui(mod, 1);
    mpz_mul_2exp(mod, mod, orderLen - 1);
    mpz_mod(sk, sk, mod);
    mpz_clear(mod);
  }

  clear_free(arr, bytes, MEM_FREE);
#endif 
}


/**
 * Isogen (sidh.h)
 * (SIDH public-key generation)
 *
 * For A:
 * Given a private key m_A, a base curve E_0, generators P_A, Q_A and P_B, Q_B:
 * - generates kernel defining an isogeny: K = P_A + m_A*Q_A
 * - gets isogenous curve E_A
 * - evaluates P_B and Q_B under the isogeny: phi_A(P_B), phi_A(Q_B)
 * - Returns public key as E_A with generators phi_A(P_B), phi_A(Q_B)
 * For B:
 * Given a private key m_B, a base curve E_0, generators P_B, Q_B and P_A, Q_A:
 * - generates kernel defining an isogeny: K = P_B + m_B*Q_B
 * - gets isogenous curve E_B
 * - evaluates P_A and Q_A under the isogeny: phi_B(P_A), phi_B(Q_A)
 * - Returns public key as E_B with generators phi_B(P_A), phi_B(Q_A)
 *
 * @param params SIDH parameters (sike_params_t: sike_params.h)
 * @param pk Public key to be generated (sike_public_key_t: montgomery.h)
 * @param sk Private key, externally provided (sike_private_key_t: sike.h)
 * @param party `ALICE` or `BOB` (Party: sike.h)
 */
template<typename env_t>
__host__ __device__ void 
sidh_isogen(env_t env,
    const sike_params_t<env_t>& params,
    sike_public_key_t<env_t>& pk,
    const typename env_t::sike_private_key_t& sk,
    const party_t party) {
  return;
#if 0
  ff_Params *p = NULL;

  unsigned long e, msb = 0;
  const mont_curve_int_t *E;
  const mont_pt_t *Po, *Qo;
  void (*iso_e)(const ff_Params *, int, const mont_curve_int_t *, mont_pt_t *,
                const mont_pt_t *, const mont_pt_t *,
                mont_curve_int_t *, mont_pt_t *, mont_pt_t *);

  if (party == ALICE) {
    p = params->EA.ffData;
    e = params->eA;
    msb = params->msbA;
    E = &params->EA;
    Po = &params->EB.P;
    Qo = &params->EB.Q;
    iso_e = iso_2_e;
  } else {
    p = params->EB.ffData;
    e = params->eB;
    msb = params->msbB - 1;
    E = &params->EB;
    Po = &params->EA.P;
    Qo = &params->EA.Q;
    iso_e = iso_3_e;
  }

  mont_curve_int_t pkInt = { 0 };
  mont_curve_init(p, &pkInt);

  fp2_Copy(p, &E->a, &pkInt.a);
  fp2_Copy(p, &E->b, &pkInt.b);

  mont_pt_t S = { 0 };
  mont_pt_init(p, &S);

  // Generate kernel
  // S:=P2+SK_2*Q2;
  mont_double_and_add(E, sk, &E->Q, &S, (int) msb);
  xADD(E, &E->P, &S, &S);

  mont_pt_copy(p, Po, &pkInt.P);
  mont_pt_copy(p, Qo, &pkInt.Q);

  iso_e(p, (int) e, &pkInt, &S, &pkInt.P, &pkInt.Q, &pkInt, &pkInt.P, &pkInt.Q);

  get_xR(p, &pkInt, pk);

  mont_pt_clear(p, &S);
  mont_curve_clear(p, &pkInt);
#endif
}


/**
 * Isoex (sidh.cu)
 * (SIDH shared secret generation)
 *
 * For A:
 * Given a private key m_A, and B's public key: curve E_B, generators phi_B(P_A), phi_B(Q_A)
 * - generates kernel defining an isogeny: K = phi_B(P_A) + m_A*phi_B(Q_A)
 * - gets isogenous curve E_AB
 * - Shared secret is the j-invariant of E_AB
 * For B:
 * Given a private key m_B, and A's public key: curve E_A, generators phi_A(P_B), phi_A(Q_B)
 * - generates kernel defining an isogeny: K = phi_A(P_B) + m_B*phi_A(Q_B)
 * - gets isogenous curve E_BA
 * - Shared secret is the j-invariant of E_BA
 *
 * @param params SIDH parameters (sike_params_t: sike_params.h)
 * @param pkO Public key of the other party (sike_public_key_t: montgomery.h)
 * @param skI Own private key (sike_private_key_t: sike.h)
 * @param party `ALICE` or `BOB` (Party: sike.h)
 * @param secret Shared secret to be generated (cgbn2_t: fp2.h)
 */
template<typename env_t>
__host__ __device__ void 
sidh_isoex(env_t env,
    const sike_params_t<env_t>& params,
    const sike_public_key_t<env_t>& pkO,
    const typename env_t::sike_private_key_t& skI,
    const party_t party,
    cgbn2_t<env_t>& secret) {
  return;
#if 0
  ff_Params *p = NULL;

  unsigned long e, msb;

  mont_curve_int_t E = { 0 };

  void (*iso_e)(const ff_Params *, int, const mont_curve_int_t *, mont_pt_t *,
                const mont_pt_t *, const mont_pt_t *,
                mont_curve_int_t *, mont_pt_t *, mont_pt_t *);

  if (party == ALICE) {
    p = params->EA.ffData;
    e = params->eA;
    msb = params->msbA;
    iso_e = iso_2_e;
  } else {
    p = params->EB.ffData;
    e = params->eB;
    msb = params->msbB - 1;
    iso_e = iso_3_e;
  }

  mont_curve_init(p, &E);
  get_yP_yQ_A_B(p, pkO, &E);

  mont_pt_t S = { 0 };
  mont_pt_init(p, &S);


  // Generate kernel
  //S:=phiP2+SK_2*phiQ2
  mont_double_and_add(&E, skI, &E.Q, &S, (int) msb);
  xADD(&E, &E.P, &S, &S);

  iso_e(p, (int) e, &E, &S, NULL, NULL, &E, NULL, NULL);

  j_inv(p, &E, secret);

  mont_curve_clear(p, &E);
  mont_pt_clear(p, &S);
#endif  
}
                
#endif // ISOGENY_REF_SIDH_H
