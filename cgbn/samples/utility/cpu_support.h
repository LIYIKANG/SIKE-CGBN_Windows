/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/

#include <stdlib.h>

#include <iostream>

#ifdef __GMP_H__
void to_mpz(mpz_t r, uint32_t *x, uint32_t count) {
  mpz_import(r, count, -1, sizeof(uint32_t), 0, 0, x);
}

void from_mpz(const mpz_t s, uint32_t *x, uint32_t count) {
  size_t words;

  if(mpz_sizeinbase(s, 2)>count*32) {
    fprintf(stderr, "from_mpz failed -- result does not fit\n");
    exit(1);
  }

  mpz_export(x, &words, -1, sizeof(uint32_t), 0, 0, s);
  while(words<count)
    x[words++]=0;
}

void from_hexStrWith0x(const char* hexStrWith0x, uint32_t* x, uint32_t count) {
  mpz_t v;
  mpz_init(v);
  int rc = mpz_set_str (v, hexStrWith0x, 0); // 0 means use of prefix
  if (rc == -1) {
    std::cerr << "invalid value " << hexStrWith0x << std::endl;
    exit(EXIT_FAILURE);
  }
  from_mpz(v, x, count);
  mpz_clear(v);
}

size_t how_many_bits(const char* hexStr) {
  mpz_t v;
  mpz_init(v);
  int rc = mpz_set_str (v, hexStr, 0); // 0 means 
  if (rc == -1) {
    std::cerr << "invalid value " << hexStr << std::endl;
    exit(EXIT_FAILURE);
  }
  size_t bitlen = mpz_sizeinbase(v, 2);
  mpz_clear(v);
  return bitlen;
}

void fp_random(uint32_t* x, const uint32_t nlimbs, const mpz_t p) {
  static gmp_randstate_t state;
  static bool isFirstCall = true;
  if (isFirstCall) {
    gmp_randinit_default(state);
    isFirstCall = false;
  }
  if (x == nullptr) {
    gmp_randclear(state);
    isFirstCall = true;
    return;
  }
  mpz_t rop; mpz_init(rop);
  mpz_urandomm (rop, state, p);
  from_mpz(rop, x, nlimbs);
  mpz_clear(rop);
}
#endif // #ifdef __GMP_H__

uint32_t random_word() {
  uint32_t random;

  random=rand() & 0xFFFF;
  random=(random<<16) + (rand() & 0xFFFF);
  return random;
}

void zero_words(uint32_t *x, uint32_t count) {
  int index;

  for(index=0;index<count;index++)
    x[index]=0;
}

void print_words(uint32_t *x, uint32_t count) {
  int index;

  for(index=count-1;index>=0;index--)
    printf("%08X", x[index]);
  printf("\n");
}

void copy_words(uint32_t *from, uint32_t *to, uint32_t count) {
  int index;

  for(index=0;index<count;index++)
    to[index]=from[index];
}

int32_t nibble(char c) {
  if('0'<=c && c<='9')
    return c-'0';
  else if('a'<=c && c<='f')
    return c-'a'+10;
  else if('A'<=c && c<='F')
    return c-'A'+10;
  else {
    printf("Invalid nibble: '%c'\n", c);
    exit(1);
  }
}

void set_words(uint32_t *x, const char *hex_string, uint32_t count) {
  int index=0, length=0, value;

  for(index=0;index<count;index++)
    x[index]=0;
  while(hex_string[length]!=0)
    length++;
  for(index=0;index<length;index++) {
    value=nibble(hex_string[length-index-1]);
    x[index/8] += value<<index%8*4;
  }
}

int compare_words(uint32_t *x, uint32_t *y, uint32_t count) {
  int index;

  for(index=count-1;index>=0;index--) {
    if(x[index]!=y[index]) {
      if(x[index]>y[index])
        return 1;
      else
        return -1;
    }
  }
  return 0;
}

void swap_words(uint32_t *a, uint32_t *b, uint32_t count) {
  uint32_t temp;
  int      index;

  for(index=0;index<count;index++) {
    temp=a[index];
    a[index]=b[index];
    b[index]=temp;
  }
}

void random_words(uint32_t *x, uint32_t count) {
  int index;

  for(index=0;index<count;index++)
    x[index]=random_word();
}
