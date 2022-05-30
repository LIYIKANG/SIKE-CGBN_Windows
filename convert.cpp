#include <string.h>
#include <stdint.h>
#include <cassert>
#include <iostream>
#include <cstring>
#include "convert.h"

#ifdef __cplusplus
extern "C" {
#endif

// string -> mpz -> array ֻ�ڱ���Ԫ�ڲ�ʹ�ã�������ʹ��
/**
* Convert the string of an interger (decimal, octal, hex) to GMP mpz_t type of the integer.
* @param x GMP mpz_t-type integer
* @param str the string of an integer in hexadecimal, binary, octal, or decimal
*/
void convert_string_to_mpz(mpz_t x,
		const std::string& str) {
	int rc = mpz_set_str(x, str.c_str(), 0);
	assert(rc == 0);
}


/**
* Convert an mpz_t-type integer to an array with at most nlimbs elements.
* @param limbs an array to be stored an integer
* @param nlimbs the number of elements in the array
* @param x an mpz_t-type integer
*/
void convert_mpz_to_array(const mpz_t x, const size_t nlimbs, uint32_t* limbs) {

	//ASSUME(mpz_sizeinbase(a, 2) <= 32*nlimbs);

	size_t used = 0;
	mpz_export(limbs, &used, -1, sizeof(uint32_t), 0, 0, x);
	// Remained limbs are initalized with all zero bits.
	while (used < nlimbs) {
		limbs[used] = 0x00000000U;
		++used;
	}
}


/**
* Convert the string of an integer to an array with at most nlimbs elements.
* @param limbs an array to be stored an integer
* @param nlimbs the number of elements in the array
* @param str a string of an integer in hexadecimal, binary, octal, or decimal
*/
 void convert_string_to_array(uint32_t* limbs,
	const size_t nlimbs,
	const std::string& str){

	mpz_t x;
	mpz_init(x);
	convert_string_to_mpz(x, str);
	convert_mpz_to_array(x, nlimbs, limbs);
	mpz_clear(x);
}


// array -> mpz -> string
/**
* Convert an array with at most nlimbs elements to the mpz_t-type integer.
* @param x an mpz_t-type integer
* @param limbs an array to be stored an integer
* @param nlimbs the number of elements in the array
*/
void convert_array_to_mpz(mpz_t x,
		const uint32_t* limbs,
		const size_t nlimbs) {

	mpz_import(x, nlimbs, -1, sizeof(uint32_t), 0, 0, limbs);
}


/**
* Convert an mpz_t-type integer to the string of the integer in the base.
* @param str a string of an integer in hexadecimal, binary, octal, or decimal
* @param base a base
* @param x an mpz_t-type integer
*/
void convert_mpz_to_string(std::string& str,
		const int32_t base,
		const mpz_t x) {

	char* s = mpz_get_str(NULL, base, x);
	str = s;
	free(s);
}


/**
* Convert an mpz_t-type integer to the string of the integer in the base.
* @param str a string of an integer in hexadecimal, binary, octal, or decimal
* @param limbs an array to be stored an integer
* @param nlimbs the number of elements in the array
*/
void convert_array_to_string(std::string& str,
		const int32_t base,
		const uint32_t* limbs,
		const size_t nlimbs) {

	mpz_t x;
	mpz_init(x);
	convert_array_to_mpz(x, limbs, nlimbs);
	convert_mpz_to_string(str, base, x);
	mpz_clear(x);
}



// Functions for outputing a big integer to the standard output
/**
* Print an integer given by an array to stdout.
* @param limbs an array to be stored an integer
* @param nlimbs the number of elements in the array
* @param base the base
*/
void output_bn_array(const uint32_t* limbs,
					const size_t nlimbs,
					const int32_t base = 16) {

	mpz_t v;
	mpz_init(v);
	convert_array_to_mpz(v, limbs, nlimbs);
	mpz_out_str(stdout, base, v);
	mpz_clear(v);
}


/**
* Output an integer given by an mpz_t type to stdout.
* @param v an integer
* @param base the base
*/
void output_mpz(const mpz_t v,
const int32_t base = 16) {
	mpz_out_str(stdout, base, v);
}

#ifdef __cplusplus
}
#endif