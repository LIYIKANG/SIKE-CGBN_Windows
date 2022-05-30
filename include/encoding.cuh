//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//

/** @file
Encodings for public keys, private keys, field-p, field-p2 and shared secrets.
*/

#ifndef ISOGENY_REF_ENCODING_H
#define ISOGENY_REF_ENCODING_H

// ================================================

// C system headers (headers in angle brackets with the .h extension), e.g., <stdlib.h>
// C++ standard library headers (without file extension), e.g., <cstddef>
// This project's .h files
#include "sike_params.cuh"
#include "sidh.cuh"

#define PUBLIC_KEY_NPS 6
#define BITS_TO_BYTES_CEIL(x) (((x) + 7) / 8)


///////////////////////////////////////////////
// Encoding functions for field-p elements
///////////////////////////////////////////////

// "os" stands for an Octal String?
// octal string to integer
__host__  inline void 
ostoi(const unsigned char* to_dec, 
    const size_t to_decLen, 
    mpz_t dec) {
  mpz_import(dec, to_decLen, -1, 1, 1, 0, to_dec);
}


// "fp" stands for feild-p element?
// octal string to an F_p element 
__host__  inline int 
ostofp(const unsigned char* to_dec, 
    const size_t np, 
    const mpz_t p, 
    mpz_t dec) {
  ostoi(to_dec, np, dec);
  int rc = mpz_cmp(dec, p) >= 0;
  return rc;
}


// Is the octal string "to_dec" equivalent to F_p^2 element "dec"?
template<typename env_t>
__host__ __device__  int 
ostofp2(env_t env,
    const unsigned char* to_dec, 
    const size_t np, 
    const mpz_t p, 
    cgbn2_t<env_t>& dec) {
  assert(false);
  return 0;
#if 0      
  int rc = 0;
  rc  = ostofp(to_dec,      np, p, dec->x0);
  rc |= ostofp(to_dec + np, np, p, dec->x1);
  return rc;
#endif  
}

// integer to octal string?
__host__ __device__  inline void 
itoos(const mpz_t to_enc, 
    unsigned char* enc) {
  mpz_export(enc, NULL, -1, 1, 1, 0, to_enc);
}


__host__ __device__  inline  void 
fptoos(const mpz_t to_enc, 
    unsigned char* enc) {
  itoos(to_enc, enc);
}


__host__ __device__  inline size_t 
get_np_len(const mpz_t p) {
  return BITS_TO_BYTES_CEIL(mpz_sizeinbase(p, 2));
}


///////////////////////////////////////////////
// Encoding functions for field-p2 elements (shared secrets)
///////////////////////////////////////////////

template<typename env_t>
__host__ __device__  void 
fp2toos(env_t env,
    const sike_params_t<env_t>& params,
    const cgbn2_t<env_t>& shared_sec,
    unsigned char* enc) {
  assert(false);
  return;
#if 0      
  size_t np = get_np_len(params->EA.ffData->mod);

  fptoos(shared_sec->x0, enc     );
  fptoos(shared_sec->x1, enc + np);
#endif
}


template<typename env_t>
__host__ __device__  void 
fp2toos_alloc(env_t env,
    const sike_params_t<env_t>& params,
    const cgbn2_t<env_t>& shared_sec,
    unsigned char** enc,
    size_t* encLen) {
  assert(false);  
#if 0      
  size_t np = get_np_len(params->EA.ffData->mod);

  *encLen = 2 * np;
  *enc = (unsigned char*)calloc(*encLen, 1);

  fp2toos(params, shared_sec, *enc);
#endif
}


template<typename env_t>
__host__ __device__  inline size_t 
fp2toos_len(env_t env,
    const sike_params_t<env_t>& params,
    const party_t party) {
  assert(false);
  return 0;
#if 0      
  if (party == ALICE) {
    return 2*get_np_len(params->EA.ffData->mod);
  } else {
    return 2*get_np_len(params->EB.ffData->mod);
  }
#endif  
}


///////////////////////////////////////////////
// Encoding/decoding functions for private keys
///////////////////////////////////////////////

template<typename env_t>
__host__ __device__  inline size_t 
sk_part_len(env_t env,
    const sike_params_t<env_t>& params,
    const party_t party) {
  assert(false);
  return 0;
#if 0      
  if (party == ALICE)
    return BITS_TO_BYTES_CEIL(params->msbA);
  else
    return BITS_TO_BYTES_CEIL(params->msbB - 1);
#endif    
}


/**
 * Decodes a private key from its octet-string-encoding.
 *
 * @param params SIDH parameters
 * @param party ALICE or BOB
 * @param sk Octet-string-encoded private key
 * @param s KEM parameter 's'. May be NULL and it won't be decoded.
 * @param skDec Decoded private key part of the octet-encoding
 * @param pkDec Decoded public key part of the octet-encoding. May be NULL and it won't be decoded.
 * @return
 */

template<typename env_t>
__host__ __device__ int 
ostosk(env_t env,
    const sike_params_t<env_t>& params,
    party_t party,
    const unsigned char* sk,
    unsigned char* s,
    sike_private_key_t<env_t>::private_key skDec,
    sike_public_key_t<env_t>& pkDec) {
  assert(false);
  return 0;
#if 0      
  size_t skLen = 0;
  int rc = 0;

  if (s != NULL)
    memcpy(s, sk, params->msg_bytes);

  skLen = sk_part_len(params, party);

  ostoi(sk + params->msg_bytes, skLen, dec);

  if (pkDec != NULL)
    rc = ostopk(params, party, sk + params->msg_bytes + skLen, pkDec);

  return rc;
#endif  
}


template<typename env_t>
__host__ __device__  void 
sktoos(env_t env,
    const sike_params_t<env_t>& params,
    const party_t party,
    const unsigned char* s,
    const sike_private_key_t<env_t>& sk,
    const sike_public_key_t<env_t>& pk,
    unsigned char* enc) {
  assert(false);
  return;
#if 0      
  size_t skPartLen = 0, np = 0;

  skPartLen = sk_part_len(params, party);

  if (party == ALICE)
    np = get_np_len(params->EA.ffData->mod);
  else
    np = get_np_len(params->EB.ffData->mod);

  memcpy(enc, s, params->msg_bytes);
  memset(enc + params->msg_bytes, 0, skPartLen + 6*np);
  itoos(sk, enc + params->msg_bytes);
  pktoos(params, pk, enc + params->msg_bytes + skPartLen, party);
#endif  
}


template<typename env_t>
__host__ __device__  size_t 
sktoos_len(env_t env,
    const sike_params_t<env_t>& params,
    const party_t party) {
  assert(false);
  return 0;
#if 0      
  size_t skLen = 0, np = 0;

  skLen = sk_part_len(params, party);

  if (party == ALICE) {
    np = get_np_len(params->EA.ffData->mod);
  } else {
    np = get_np_len(params->EB.ffData->mod);
  }
  return params->msg_bytes + skLen + PUBLIC_KEY_NPS*np;
#endif  
}


///////////////////////////////////////////////
// Encoding functions for public keys
///////////////////////////////////////////////

template<typename env_t>
__host__ __device__  int 
ostopk(env_t env,
  const sike_params_t<env_t>& params,
  party_t party,
  const unsigned char *pk,
  sike_public_key_t<env_t>& dec) {
  assert(false);
  return 0;
#if 0
  int rc = 0;

  size_t np = (party == ALICE ?
               get_np_len(params->EA.ffData->mod) :
               get_np_len(params->EB.ffData->mod));

  rc  = ostofp2(pk       , np, params->p, &dec->xP);
  rc |= ostofp2(pk + 2*np, np, params->p, &dec->xQ);
  rc |= ostofp2(pk + 4*np, np, params->p, &dec->xR);

  dec->ffData = (party == ALICE ? params->EA.ffData : params->EB.ffData);

  return rc;
#endif  
}


template<typename env_t>
__host__ __device__  void 
pktoos(env_t env,
    const sike_params_t<env_t>& params,
    const sike_public_key_t<env_t>& pk,
    unsigned char* enc,
    party_t party) {
  assert(false);
  return;
#if 0      
  size_t np = (party == ALICE ?
               get_np_len(params->EA.ffData->mod) :
               get_np_len(params->EB.ffData->mod));

  memset(enc, 0, PUBLIC_KEY_NPS*np);

  fp2toos(params, &pk->xP, enc       );
  fp2toos(params, &pk->xQ, enc + 2*np);
  fp2toos(params, &pk->xR, enc + 4*np);
#endif  
}


template<typename env_t>
__host__ __device__  void 
pktoos_alloc(env_t env,
    const sike_params_t<env_t>& params,
    const sike_public_key_t<env_t>* pk,
    unsigned char **enc,
    size_t *encLen,
    party_t party) {
  assert(false);
  return;
#if 0      
  *encLen = pktoos_len(params, party);
  *enc = (unsigned char*)calloc(*encLen, 1);
  pktoos(params, pk, *enc, party);
#endif
}


template<typename env_t>
__host__ __device__  size_t 
pktoos_len(env_t env,
    const sike_params_t<env_t>& params,
    const party_t party) {
  assert(false);
  return 0;
#if 0      
  if (party == ALICE)
    return PUBLIC_KEY_NPS * get_np_len(params->EA.ffData->mod);
  else
    return PUBLIC_KEY_NPS * get_np_len(params->EB.ffData->mod);
#endif    
}


///////////////////////////////////////////////
// Encoding functions for encapsulations (ct)
///////////////////////////////////////////////
/**
 * Octet-string-to-encapsulation conversion (decoding)
 *
 * @param params SIKE parameters
 * @param ct Octet-encoded encapsulation to be decoded.
 * @param c0 Decoded parameter
 * @param c1 Decoded parameter
 * @return
 */

template<typename env_t>
__host__ __device__ int 
ostoencaps(env_t env,
    const sike_params_t<env_t>& params,
    const unsigned char* ct,
    sike_public_key_t<env_t>& c0,
    unsigned char* c1) {
  assert(false);
  return 0;
#if 0      
  int rc = 0;
  size_t np = get_np_len(params->EA.ffData->mod);

  rc = ostopk(params, BOB, ct, c0);
  memcpy(c1, ct + PUBLIC_KEY_NPS*np, params->msg_bytes);

  return rc;
#endif  
}


template<typename env_t>
__host__ __device__  void 
encapstoos(env_t env,
    const sike_params_t<env_t>& params,
    const sike_public_key_t<env_t>& c0,
    const unsigned char* c1,
    unsigned char* ct) {
  assert(false);
  return;

#if 0
  size_t np = get_np_len(params->EB.ffData->mod);

  memset(ct, 0, PUBLIC_KEY_NPS*np);

  pktoos(params, c0, ct, BOB);

  memcpy(ct + PUBLIC_KEY_NPS*np, c1, params->msg_bytes);
#endif  
}


template<typename env_t>
__host__ __device__  void 
encapstoos_alloc(env_t env,
    const sike_params_t<env_t>& params,
    const sike_public_key_t<env_t>* c0,
    const unsigned char* c1,
    unsigned char** ct,
    size_t* ctLen) {
  assert(false);
  return;
#if 0      
  *ctLen = encapstoos_len(params);
  *ct = (unsigned char*)calloc(*ctLen, 1);

  encapstoos(params, c0, c1, *ct);
#endif  
}


template<typename env_t>
__host__ __device__  size_t 
encapstoos_len(env_t env,
    const sike_params_t<env_t>& params) {
  assert(false);
  return 0;
#if 0      
  int components = PUBLIC_KEY_NPS;

  size_t np = get_np_len(params->EB.ffData->mod);
  return components * np + params->msg_bytes;
#endif  
}


// Memory functions
__host__ __device__ inline void
clear_free(void* ptr, 
  const size_t size, 
  const int free_mem) {
  if (ptr) {
    memset(ptr, 0, size);
    *(volatile unsigned char*)ptr = *(volatile unsigned char*)ptr;
  //  if (free_mem == MEM_FREE)
  //    free(ptr);
  }
}

#endif //ISOGENY_REF_ENCODING_H
