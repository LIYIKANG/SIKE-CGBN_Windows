//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//

#ifndef ISOGENY_REF_SIDH_API_GENERIC_H
#define ISOGENY_REF_SIDH_API_GENERIC_H

#include "sike_params.cuh"
#include "encoding.cuh"
#include "sidh.cuh"

// SIKE's key generation. Generic version, Supports all parameter sets.
// It produces a private key sk and computes the public key pk.
// Outputs: secret key sk
//          public key pk
template<typename env_t>
__host__ __device__ int 
crypto_kem_keypair_generic(env_t env,
    const sike_params_t<env_t>& params, 
    unsigned char* pk, 
    unsigned char* sk){
    
    ff_Params<env_t>& p = params.EB.ffData;//params->EB.ffData;

    sike_private_key_t<env_t>::private_key sk3;  
    sike_public_key_t<env_t> pk3; //{ 0 };
    unsigned char s[params->msg_bytes];

    fp_Init(env, p, sk3);
    public_key_init(env, p, pk3);

    sike_kem_keygen(env, params, pk3, sk3, s);

    pktoos(env, params, pk3, pk, BOB);
    sktoos(env, params, BOB, s, sk3, pk3, sk);

    fp_Clear(env, p, sk3);
    public_key_clear(env, p, pk3);
    return 0;

}


// SIKE's encapsulation Generic version, Supports all parameter sets.
// Input:   public key pk
// Outputs: shared secret ss
//          ciphertext message ct (this is c0 and c1 encoded)
template<typename env_t>
__host__ __device__ int 
crypto_kem_enc_generic(env_t env,
    const sike_params_t<env_t>& params,
    unsigned char* ct,
    unsigned char* ss,
    const unsigned char* pk){

  int rc = 0;
  
  ff_Params<env_t>& p = params.EB.ffData;
  sike_public_key_t<env_t> c0;  //{ 0 };
  sike_public_key_t<env_t> pk3; //{ 0 };
  unsigned char c1[params->msg_bytes];

  public_key_init(env, p, c0); 
  public_key_init(env, p, pk3);

  rc = ostopk(env, params, BOB, pk, pk3);
  if ( rc ) goto end;

  sike_kem_encaps(env, params, pk3, c0, c1, ss);

  encapstoos(env, params, c0, c1, ct);

end:
  public_key_clear(env, p, c0);
  public_key_clear(env, p, pk3);
  return rc;
}


// SIKE's decapsulation
// Input:   secret key sk
//          ciphertext message ct
// Outputs: shared secret ss
template<typename env_t>
__host__ __device__ int
crypto_kem_dec_generic(env_t env,
    const sike_params_t<env_t>& params,
    unsigned char* ss,
    const unsigned char* ct,
    const unsigned char* sk){

  ff_Params<env_t>& p = params.EB.ffData;
  sike_private_key_t<env_t>::private_key sk3; 
  sike_public_key_t<env_t> pk3;  //{ 0 };
  sike_public_key_t<env_t> c0;   //{ 0 };
  unsigned char s[params->msg_bytes];
  unsigned char c1[params->msg_bytes];

  fp_Init(env, p, sk3);
  public_key_init(env, p, pk3);
  public_key_init(env, p, c0);

  ostosk(env, params, BOB, sk, s, sk3, pk3);

  ostoencaps(env, params, ct, c0, c1);

  sike_kem_decaps(env, params, pk3, sk3, c0, c1, s, ss);

  fp_Clear(env, p, sk3);
  public_key_clear(env, p, pk3);
  public_key_clear(env, p, c0);
  return 0;
}

#endif //ISOGENY_REF_SIDH_API_GENERIC_H

// end of file
