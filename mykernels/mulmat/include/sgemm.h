#pragma once
#include <stdint.h>
#include <stdbool.h>

#if defined(__VXE__) || defined(__VXE2__)
#include <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
extern "C" {
#endif


bool llamafile_sgemm(int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     int, int, int);


#if defined(PACC)
void mul_mat_f16(int m, int n, int k,
                 const uint16_t *A, int lda, // lda in bytes
                 const uint16_t *B, int ldb, // ldb in bytes
                 float *C, int ldc);

void mul_mat_bf16(int m, int n, int k,
                  const uint16_t *A, int lda, // lda in bytes
                  const uint16_t *B, int ldb, // ldb in bytes
                  float *C, int ldc);
#endif


#ifdef __cplusplus
}
#endif
