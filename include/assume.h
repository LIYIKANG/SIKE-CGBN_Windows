/* ASSUME() works same as standard-function assert().
   This function ignores NODEBUG macro, unlikely assert(). */
#ifndef INCLUDE_ASSUME_H_
#define INCLUDE_ASSUME_H_


#if !defined(ASSUME)
  #include <cassert>
  #define ASSUME(expr) \
    ((expr) ? ((void)(0)) : __assert_fail(#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))
#else
  #error already defined: ASSUME()
#endif // !defined(ASSUME)

#endif // INCLUDE_ASSUME_H_

// end of file
