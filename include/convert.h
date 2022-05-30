#ifndef INCLUDE_CONVERT_H_
#define INCLUDE_CONVERT_H_

#include <string.h>
#include <stdint.h>
#include "gmp.h"

#ifdef __cplusplus
extern "C" {
#endif

void convert_string_to_mpz(mpz_t x,
		const std::string& str);

void convert_mpz_to_array(const mpz_t x, const size_t nlimbs, uint32_t* limbs);

void convert_string_to_array(uint32_t* limbs,	const size_t nlimbs, const std::string& str);

void convert_array_to_mpz(mpz_t x,	const uint32_t* limbs,	const size_t nlimbs);

void convert_mpz_to_string(std::string& str, const int32_t base, const mpz_t x);

void convert_array_to_string(std::string& str,	const int32_t base,	const uint32_t* limbs,	const size_t nlimbs);


void output_bn_array(const uint32_t* limbs,	const size_t nlimbs, const int32_t base /* = 16*/);

void output_mpz(const mpz_t v, const int32_t base /*= 16*/);


#ifdef __cplusplus
}
#endif



template<class env_t>
void convert_array_to_cgbn(env_t env, const uint32_t* limbs, const size_t nlimbs, typename env_t::cgbn_t& a){
	// Assume that "limbs" is produced by exporting an mpz_t variable
	for (size_t i = 0; i < nlimbs; ++i) {
		a._limbs[i] = limbs[i];
	}
}

#endif // INCLUDE_CONVERT_H_

// end of file
