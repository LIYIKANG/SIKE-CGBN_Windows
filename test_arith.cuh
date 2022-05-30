//
// Supersingular Isogeny Key Encapsulation Ref. Library
//
// InfoSec Global Inc., 2017-2020
// Basil Hess <basil.hess@infosecglobal.com>
//

#ifndef ISOGENY_REF_TEST_ARITH_H
#define ISOGENY_REF_TEST_ARITH_H


struct sike_params_raw_t;

/**
 * Tests Fp and Fp2 arithmetic with the given SIDH parameters
 * @param params SIDH parameters
 * @return 0 if tests succeeded, 1 otherwise
 */
bool 
test_arith(const sike_params_raw_t& params_raw);

#endif //ISOGENY_REF_TEST_ARITH_H

// end of file
