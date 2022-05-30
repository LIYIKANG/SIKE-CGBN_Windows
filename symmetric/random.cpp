/********************************************************************************************
* Hardware-based random number generation function using /dev/urandom
*********************************************************************************************/

#include <random.h>
#include <stdlib.h>
#include <fcntl.h>
#include "unistd.h"
static int lock = -1;


static __inline void delay(unsigned int count)
{
  while (count--) {}
}


#ifdef __cplusplus
extern "C" {
#endif

//todo: 此函数改成windows版本 和rng.h重复了

//int randombytes(unsigned char* random_array, unsigned long long xlen)
//{}

#ifdef __cplusplus
}
#endif