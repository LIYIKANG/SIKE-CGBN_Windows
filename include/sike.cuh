//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//

/** @file
SIKE KEM and PKE functions:
 PKE encryption
 PKE decryption
 KEM encapsulation
 KEM decapsulation
*/

#ifndef ISOGENY_REF_SIKE_H
#define ISOGENY_REF_SIKE_H


// ================================================
// C system headers (headers in angle brackets with the .h extension), e.g., <stdlib.h>
// C++ standard library headers (without file extension), e.g., <cstddef>
// Other libraries' .h files
// This project's .h files
#include "fips202.h"
#include "sike_params.cuh"
#include "fp.cuh"


/**
 * ALICE or BOB (sike.h)
 */
//enum class Party { Alice, Bob, Unknown };
typedef enum {ALICE, BOB} party_t;

/**
 * A private key can be represented by a multi-precision value. (sike.h)
 */
template<typename env_t>
struct sike_private_key_t {
  typename env_t::cgbn_t private_key;
};


/**
 * A message m (sike.h)
 */
typedef unsigned char sike_msg;


/**
 * Function f: SHAKE256 (sike.h)
 */
__host__ static void 
function_F(const unsigned char* m, 
    const size_t mLen, 
    unsigned char* F, 
    size_t fLen) {
  shake256(F, fLen, m, mLen);
}


/**
 * Function g: SHAKE256 (sike.h)
 */
__host__ static void 
function_G(const unsigned char* m, 
    const size_t mLen, 
    unsigned char* G, 
    size_t gLen) {
  shake256(G, gLen, m, mLen);
}


/**
 * Function h: SHAKE256 (sike.h)
 */
__host__ static void 
function_H(const unsigned char* m, 
    const size_t mLen, 
    unsigned char* H, 
    size_t hLen) {
  shake256(H, hLen, m, mLen);
}

template<typename env_t>
struct sike_params_t;


template<typename env_t>
struct sike_public_key_t;
/**
 * SIKE PKE encryption (sike.h)
 *
 * For B:
 * - c_0 == PK_3 <- B's keygen function using PK_3, SK_3
 * - j <- Shared secret (j-invariant) using PK_2, SK_3
 * - h <- F(j)
 * - c_1 <- h + m
 * - return (c_0, c_1)
 * -
 * @param params SIKE parameters (sike_params_t: sike_params.h)
 * @param pk3 Public key of the other party (sike_public_key_t: montgomery.h)
 * @param m message (sike_msg*: sike.h)
 * @param sk2 Own private key (sike_private_key: sike.h)
 * @param c0 First component of encryption (sike_public_key_t: montgomery.h)
 * @param c1 Second component of encryption (unsigned char*)
 */
template<typename env_t>
__host__ __device__ void
sike_pke_enc(env_t env,
    const sike_params_t<env_t>& params,
    const sike_public_key_t<env_t>& pk3,
    const sike_msg* m,
    const typename env_t::sike_private_key sk2,
    sike_public_key_t<env_t>& c0,
    unsigned char* c1) {
  return;
#if 0
  ff_Params* p = params->EB.ffData;

  fp2 j = { 0 };
  unsigned char* jEnc = NULL;
  size_t jEncLen = 0;
  unsigned char h[params->msg_bytes];

  fp2_Init(p, &j);

  // c0 <- isogen_2(sk_2)
  sidh_isogen(params, c0, sk2, ALICE);

  // j <- isoex_2(pk3, sk2)
  sidh_isoex(params, pk3, sk2, ALICE, &j);

  fp2toos_alloc(params, &j, &jEnc, &jEncLen);

  // h <- F(j)
  function_F(jEnc, jEncLen, h, params->msg_bytes);

  // c1 <- h ^ m
  for (int i = 0; i < params->msg_bytes; ++i)
    c1[i] = h[i] ^ m[i];

  // cleanup
  clear_free(h, params->msg_bytes, MEM_NOT_FREE);
  fp2_Clear(p, &j);
  clear_free(jEnc, jEncLen, MEM_FREE);
#endif  
}
    

/**
 * SIKE PKE decryption (sike.h)
 *
 * For B:
 * - B's keygen function using PK_2, SK_3, evaluating on B's curve
 * - Shared secret (j-invariant),
 *
 * @param params SIKE parameters (sike_params_t: sike_params.h)
 * @param sk3 Own private key (sike_private_key_t: sike.h)
 * @param c0 First component of encryption (sike_public_key_t: montgomery.h)
 * @param c1 Second component of encryption (usigned char*)
 * @param m Recovered message (sike_msg*: sike.h)
 */
template<typename env_t>
__host__ __device__ void
sike_pke_dec(env_t env,
    const sike_params_t<env_t>& params,
    const typename env_t::sike_private_key_t& sk3,
    const sike_public_key_t<env_t>& c0,
    const unsigned char* c1,
    sike_msg* m) {
  return;
#if 0      
  const ff_Params* p = params->EA.ffData;
  fp2 j = { 0 };
  unsigned char h[params->msg_bytes];
  unsigned char* jEnc = NULL;
  size_t jEncLen = 0;

  fp2_Init(p, &j);

  // j <- isoex_3(c0, sk3)
  sidh_isoex(params, c0, sk3, BOB, &j);

  fp2toos_alloc(params, &j, &jEnc, &jEncLen);

  // h <- F(j)
  function_F(jEnc, jEncLen, h, params->msg_bytes);

  // c1 = h ^ m
  for (int i = 0; i < params->msg_bytes; ++i)
    m[i] = h[i] ^ c1[i];

  // cleanup
  clear_free(h, params->msg_bytes, MEM_NOT_FREE);
  fp2_Clear(p, &j);
  clear_free(jEnc, jEncLen, MEM_FREE);
#endif  
}
    

/**
 * SIKE KEM key generation (KeyGen) (sike.h)
 *
 * @param params SIKE parameters (sike_params_t: sike_params.h)
 * @param pk3 public key (sike_public_key_t: montgomery.h)
 * @param sk3 private key (sike_private_key_t: sike.h)
 * @param s SIKE parameter s (unsigned char*)
 */
template<typename env_t>
__host__ __device__ void
sike_kem_keygen(env_t env,
    const sike_params_t<env_t>& params,
    sike_public_key_t<env_t>& pk3,
    typename env_t::sike_private_key& sk3,
    unsigned char *s) {
  return;
#if 0      
  randombytes(s, params->msg_bytes);

  sidh_sk_keygen(params, BOB, sk3);
  sidh_isogen(params, pk3, sk3, BOB);
#endif  
}
    

/**
 * SIKE KEM Encapsulation (sike.h)
 *
 * For B:
 * - m <- random(0,1)^l
 * - r <- G(m || pk3)
 * - (c0, c1) <- Enc(pk3, m, r)
 * - K <- H(m || (c0, c1))
 *
 * @param params SIKE parameters (sike_params_t: sike_params.h)
 * @param pk3 Other party's public key (sike_public_key_t: montgomery.h)
 * @param c0 First component of encryption (sike_public_key_t: montgomery.h)
 * @param c1 Second component of encryption (unsigned char*)
 * @param K key (do not share with other party) (unsigned char*)
 */
template<typename env_t>
__host__ __device__ void
sike_kem_encaps(env_t env,
    const sike_params_t<env_t>& params,
    const sike_public_key_t<env_t>& pk3,
    sike_public_key_t<env_t>& c0,
    unsigned char* c1,
    unsigned char* K) {
  return;
#if 0
  const ff_Params* p = params->EA.ffData;
  size_t rLen = BITS_TO_BYTES_CEIL(params->msbA);
  unsigned char* r = (unsigned char*)calloc(rLen, 1);

  sike_private_key rDec;
  fp_Init(p, rDec);

  // space for (m || pk3)
  size_t mPk3EncLen = params->msg_bytes + pktoos_len(params, BOB);
  unsigned char mPk3Enc[mPk3EncLen];
  unsigned char* m = mPk3Enc, *pk3Enc = mPk3Enc + params->msg_bytes;

  // m <- random(0,1)^l
  randombytes(m, params->msg_bytes);

  pktoos(params, pk3, pk3Enc, BOB);

  // r <- G(m || pk3)
  function_G(mPk3Enc, mPk3EncLen, r, rLen);


  ostoi(r, BITS_TO_BYTES_CEIL(params->msbA), rDec);
  mp_mod(rDec, params->ordA, rDec);

  // (c0, c1) <- Enc(pk3, m, r)
  sike_pke_enc(params, pk3, m, rDec, c0, c1);

  size_t mCEncLen = params->msg_bytes + encapstoos_len(params);
  unsigned char mCEnc[mCEncLen];
  unsigned char* cEnc = mCEnc + params->msg_bytes;
  memcpy(mCEnc, m, params->msg_bytes);
  encapstoos(params, c0, c1, cEnc);

  // K <- H(m || (c0, c1))
  function_H(mCEnc, mCEncLen, K, params->crypto_bytes);

  // cleanup
  clear_free(r, rLen, MEM_FREE);
  fp_Clear(p, rDec);
#endif  
}
    

/**
 * SIKE KEM Decapsulation (sike.h)
 *
 * For B:
 * - m'  <- Dec(sk3, (c0, c1))
 * - r'  <- G(m' || pk3)
 * - c0' <- isogen_2(r')
 * - if (c0' == c0) K <- H(m' || (c0, c1))
 * - else           K <- H(s || (c0, c1))
 * 
 * @param params SIKE parameters (sike_params_t: sike_params.h)
 * @param pk3 Own public key (sike_public_key_t: montgomery.h)
 * @param sk3 Own private key (sike_private_key_t: sike.h)
 * @param c0 First component of the encryption (sike_public_key_t: montgomery.h)
 * @param c1 Second component of the encrytion (unsigned char*)
 * @param s SIKE parameter `s` (unsigned char*)
 * @param K decapsulated keys (unsigned char*)
 */
template<typename env_t>
__host__ __device__ void
sike_kem_decaps(env_t env,
    const sike_params_t<env_t>& params,
    const sike_public_key_t<env_t>& pk3,
    const sike_private_key_t<env_t>::private_key sk3,
    const sike_public_key_t<env_t>& c0,
    const unsigned char* c1,
    const unsigned char* s,
    unsigned char* K) {
  return;
#if 0      
  ff_Params* p = params->EA.ffData;

  // space for (m || pk3)
  size_t mPk3EncLen = params->msg_bytes + pktoos_len(params, BOB);
  unsigned char mPk3Enc[mPk3EncLen];
  unsigned char* m = mPk3Enc, *pk3Enc = mPk3Enc + params->msg_bytes;
  memset(m, 0, params->msg_bytes);

  // m' <- Dec(sk3, (c0, c1))
  sike_pke_dec(params, sk3, c0, c1, m);

  size_t rLen = BITS_TO_BYTES_CEIL(params->msbA);
  unsigned char* r = (unsigned char*)calloc(rLen, 1);

  sike_private_key rDec;
  fp_Init(p, rDec);

  pktoos(params, pk3, pk3Enc, BOB);

  // r' <- G(m' || pk3)
  function_G(mPk3Enc, mPk3EncLen, r, rLen);

  ostoi(r, rLen, rDec);
  mp_mod(rDec, params->ordA, rDec);

  sike_public_key_t c0Prime = { 0 };
  public_key_init(p, &c0Prime);

  // c0' <- isogen_2(r')
  sidh_isogen(params, &c0Prime, rDec, ALICE);

  unsigned char *c0PrimeEnc = NULL, *c0Enc = NULL;
  size_t c0PrimeEncLen = 0, c0EncLen = 0;
  pktoos_alloc(params, &c0Prime, &c0PrimeEnc, &c0PrimeEncLen, ALICE);
  pktoos_alloc(params, c0, &c0Enc, &c0EncLen, ALICE);

  size_t mCEncLen = params->msg_bytes + encapstoos_len(params);
  unsigned char mCEnc[mCEncLen];
  unsigned char* cEnc = mCEnc + params->msg_bytes;
  encapstoos(params, c0, c1, cEnc);

  int eq = !memcmp(c0Enc, c0PrimeEnc, c0PrimeEncLen);
  if (eq) {
    // K <- H(m' || (c0, c1))
    memcpy(mCEnc, m, params->msg_bytes);
    function_H(mCEnc, mCEncLen, K, params->crypto_bytes);
  } else {
    // K <- H(s || (c0, c1))
    memcpy(mCEnc, s, params->msg_bytes);
    function_H(mCEnc, mCEncLen, K, params->crypto_bytes);
  }

  // cleanup
  clear_free(r, rLen, MEM_FREE);
  fp_Clear(p, rDec);
  public_key_clear(p, &c0Prime);
  clear_free(c0PrimeEnc, c0PrimeEncLen, MEM_FREE);
  clear_free(c0Enc, c0EncLen, MEM_FREE);
  clear_free(mPk3Enc, mPk3EncLen, MEM_NOT_FREE);
#endif  
}

#endif //ISOGENY_REF_SIKE_H
