#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "pacc.h"

#include "common.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-threading.h"
#include "ggml.h"
#include "simd-mappings.h"
#include "traits.h"

#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>  // for GGML_ASSERT
#include <stdexcept>
#include <thread>
#include <atomic>
#if defined(__gnu_linux__)
#    include <syscall.h>
#endif


// clang-format off
#if defined(__riscv)

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#else
#include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh)
#error "riscv zfh extension not enabled"
#endif

#else

#error "riscv not enabled in this build"

#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#define UNUSED GGML_UNUSED

#define QGEMM_STRIDEN_THREAD_ALIGN 32
#define CACHE_LINE_SIZE 64

// clang-format on

template <int layout_block_n>
static inline void micro_kernel_fp16fp16fp32_tile_k1_tile_n_gemv(int              m_v,
                                                                 int              n_v,
                                                                 int              k_v,
                                                                 const _Float16 * A,
                                                                 const _Float16 * B,
                                                                 float *          C) {
    const _Float16 * b_ptr = B;
    int kk = 0;

    size_t vl = __riscv_vsetvl_e32m8(MIN(n_v, layout_block_n));

    vfloat32m8_t sumf = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    for (; kk + 3 < k_v; kk += 4) {
        _Float16 a0 = A[kk + 0];
        _Float16 a1 = A[kk + 1];
        _Float16 a2 = A[kk + 2];
        _Float16 a3 = A[kk + 3];

        vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(b_ptr, vl);
        b_ptr += vl;
        vfloat16m4_t vb1 = __riscv_vle16_v_f16m4(b_ptr, vl);
        b_ptr += vl;
        vfloat16m4_t vb2 = __riscv_vle16_v_f16m4(b_ptr, vl);
        b_ptr += vl;
        vfloat16m4_t vb3 = __riscv_vle16_v_f16m4(b_ptr, vl);
        b_ptr += vl;

        sumf = __riscv_vfwmacc_vf_f32m8_tu(sumf, a0, vb0, vl);
        sumf = __riscv_vfwmacc_vf_f32m8_tu(sumf, a1, vb1, vl);
        sumf = __riscv_vfwmacc_vf_f32m8_tu(sumf, a2, vb2, vl);
        sumf = __riscv_vfwmacc_vf_f32m8_tu(sumf, a3, vb3, vl);
    }

    for (; kk < k_v; ++kk) {
        _Float16 a0 = A[kk];
        vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(b_ptr, vl);
        sumf = __riscv_vfwmacc_vf_f32m8_tu(sumf, a0, vb0, vl);
        b_ptr += vl;
    }

    __riscv_vse32_v_f32m8(C, sumf, vl);
}

template <int layout_block_n>
static inline void micro_kernel_bf16bf16fp32_tile_k1_tile_n_gemv(int            m_v,
                                                                 int            n_v,
                                                                 int            k_v,
                                                                 const __bf16 * A,
                                                                 const __bf16 * B,
                                                                 float *        C) {
    const __bf16 * b_ptr = B;
    int kk = 0;

    size_t vl = __riscv_vsetvl_e32m8(MIN(n_v, layout_block_n));

    vfloat32m8_t sumf = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    for (; kk + 3 < k_v; kk += 4) {
        __bf16 a0 = A[kk + 0];
        __bf16 a1 = A[kk + 1];
        __bf16 a2 = A[kk + 2];
        __bf16 a3 = A[kk + 3];

        vbfloat16m4_t vb0 = __riscv_vle16_v_bf16m4(b_ptr, vl);
        b_ptr += vl;
        vbfloat16m4_t vb1 = __riscv_vle16_v_bf16m4(b_ptr, vl);
        b_ptr += vl;
        vbfloat16m4_t vb2 = __riscv_vle16_v_bf16m4(b_ptr, vl);
        b_ptr += vl;
        vbfloat16m4_t vb3 = __riscv_vle16_v_bf16m4(b_ptr, vl);
        b_ptr += vl;

        sumf = __riscv_vfwmaccbf16_vf_f32m8_tu(sumf, a0, vb0, vl);
        sumf = __riscv_vfwmaccbf16_vf_f32m8_tu(sumf, a1, vb1, vl);
        sumf = __riscv_vfwmaccbf16_vf_f32m8_tu(sumf, a2, vb2, vl);
        sumf = __riscv_vfwmaccbf16_vf_f32m8_tu(sumf, a3, vb3, vl);
    }

    for (; kk < k_v; ++kk) {
        __bf16 a0 = A[kk];
        vbfloat16m4_t vb0 = __riscv_vle16_v_bf16m4(b_ptr, vl);
        sumf = __riscv_vfwmaccbf16_vf_f32m8_tu(sumf, a0, vb0, vl);
        b_ptr += vl;
    }

    __riscv_vse32_v_f32m8(C, sumf, vl);    
}

template <int layout_block_n>
static void ggml_compute_forward_mul_mat_one_chunk(const struct ggml_compute_params * params,
                                                   struct ggml_tensor *               dst,
                                                   const enum ggml_type               type,
                                                   const int64_t                      num_rows_per_vec_dot,
                                                   const int64_t                      ir0_start,
                                                   const int64_t                      ir0_end,
                                                   const int64_t                      ir1_start,
                                                   const int64_t                      ir1_end) {
    constexpr int block_n = layout_block_n;
    constexpr int block_m = 1;

    //{type = GGML_TYPE_Q8_0, buffer = 0x4b62a90, ne = {1024, 6144, 1, 1}, nb = {34, 1088, 6684672, 6684672}, op = GGML_OP_NONE,
    const struct ggml_tensor * src0 = dst->src[0];

    //type = GGML_TYPE_F32, buffer = 0x91a180, ne = {1024, 4, 1, 1}, nb = {4, 4096, 16384, 16384}, op = GGML_OP_MUL,
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);
    const bool dst_cont  = ggml_is_contiguous(dst);

    assert(ir0_start % block_n == 0);

    assert(src1_cont == true);
    assert(dst_cont == true);
    
    enum ggml_type const vec_dot_type = ggml_get_type_traits_cpu(type)->vec_dot_type;

    bool is_fp16_type = vec_dot_type == GGML_TYPE_F16;
    bool is_bf16_type = vec_dot_type == GGML_TYPE_BF16;
    bool is_q8_0_type = vec_dot_type == GGML_TYPE_Q8_0;
    assert(is_fp16_type || is_bf16_type || is_q8_0_type);

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;

    const char * new_src0 = (const char *) src0->data;                          //weight
    if (strncmp(src0->name, "token_embd.weight", 17) == 0)
      new_src0 += src0->nb[2];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += block_m) {
        const char * src0_row = new_src0 + (ir0_start * src0->nb[1]);                          //weight
        const char * src1_col = (const char *) wdata + (iir1 * row_size);                                       //active
        float * dst_col  = (float *) ((char *) dst->data + (iir1 * dst->nb[1]) + (ir0_start * sizeof(float)));  //result
        size_t  tile_k_v = ne00;
        size_t  tile_n_v = ir0_end - ir0_start;
        size_t  tile_m_v = MIN(ir1_end - iir1, block_m);
        if (is_bf16_type) {
            micro_kernel_bf16bf16fp32_tile_k1_tile_n_gemv<block_n>(tile_m_v, tile_n_v, tile_k_v, (__bf16 *) src1_col,
                                                                   (__bf16 *) src0_row, dst_col);
        } else if (is_fp16_type) {
            micro_kernel_fp16fp16fp32_tile_k1_tile_n_gemv<block_n>(tile_m_v, tile_n_v, tile_k_v, (_Float16 *) src1_col,
                                                                   (_Float16 *) src0_row, dst_col);
        } else {
            assert(0);
        }
    }
}

namespace ggml::cpu::riscv64_pacc {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(struct ggml_tensor *, const void *, size_t);


class tensor_traits_base : public ggml::cpu::tensor_traits {
  public:
    virtual int repack(struct ggml_tensor * t, const void * data, size_t data_size) = 0;
};

/* Determines the optimal vectorization strategy for 16-bit element transpose.
 * Returns true for M-dimension vectorization, false for N-dimension
 * vectorization.
 */
bool pacc_transpose_e16_is_mvec(size_t m, size_t n) {
  size_t vlmax = __riscv_vsetvlmax_e16m1();
  if (m > n || m >= vlmax) {
    return true;
  }
  return false;
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using strided segment-8 loads and unit-strided
 * stores. Vectorization is performed along the M dimension, and N is expected
 * to be a multiple of 8.
 */
void
pacc_transpose_mvec_x8_e16_zve32x(size_t m, size_t n,
                                 const uint16_t *GGML_RESTRICT a, size_t rsa,
                                 uint16_t *GGML_RESTRICT at, size_t rsat) {

  const ptrdiff_t input_seg_bstride = (ptrdiff_t)(rsa * sizeof(uint16_t));

  size_t vl = 0;
  for (size_t ii = 0; ii + 7 < n; ii += 8) {
    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint16m1x8_t vcols = __riscv_vundefined_u16m1x8();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)

    const uint16_t *in_tile = a + ii;
    uint16_t *out_tile = at + ii * rsat;
    for (size_t jj = 0; jj  < m; jj += vl) {
      vl = __riscv_vsetvl_e16m1(m - jj);

      vcols = __riscv_vlsseg8e16_v_u16m1x8(in_tile, input_seg_bstride, vl);

      uint16_t *write_ptr = out_tile;
      const size_t output_seg_stride = rsat;
      // store each segment continuously along m
      __riscv_vse16_v_u16m1(write_ptr, __riscv_vget_v_u16m1x8_u16m1(vcols, 0),
                            vl);
      write_ptr += output_seg_stride;
      __riscv_vse16_v_u16m1(write_ptr, __riscv_vget_v_u16m1x8_u16m1(vcols, 1),
                            vl);
      write_ptr += output_seg_stride;
      __riscv_vse16_v_u16m1(write_ptr, __riscv_vget_v_u16m1x8_u16m1(vcols, 2),
                            vl);
      write_ptr += output_seg_stride;
      __riscv_vse16_v_u16m1(write_ptr, __riscv_vget_v_u16m1x8_u16m1(vcols, 3),
                            vl);
      write_ptr += output_seg_stride;
      __riscv_vse16_v_u16m1(write_ptr, __riscv_vget_v_u16m1x8_u16m1(vcols, 4),
                            vl);
      write_ptr += output_seg_stride;
      __riscv_vse16_v_u16m1(write_ptr, __riscv_vget_v_u16m1x8_u16m1(vcols, 5),
                            vl);
      write_ptr += output_seg_stride;
      __riscv_vse16_v_u16m1(write_ptr, __riscv_vget_v_u16m1x8_u16m1(vcols, 6),
                            vl);
      write_ptr += output_seg_stride;
      __riscv_vse16_v_u16m1(write_ptr, __riscv_vget_v_u16m1x8_u16m1(vcols, 7),
                            vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using strided segment-4 loads and unit-strided
 * stores. Vectorization is performed along the M dimension, and N is expected
 * to be a multiple of 4.
 */
void
pacc_transpose_mvec_x4_e16_zve32x(size_t m, size_t n,
                                 const uint16_t *GGML_RESTRICT a, size_t rsa,
                                 uint16_t *GGML_RESTRICT at, size_t rsat) {

  const ptrdiff_t input_seg_bstride = (ptrdiff_t)(rsa * sizeof(uint16_t));

  size_t vl = 0;

  for (size_t ii = 0; ii + 3 < n; ii += 4) {

    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint16m2x4_t vcols = __riscv_vundefined_u16m2x4();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)
    const uint16_t *in_tile = a + ii;
    uint16_t *out_tile = at + ii * rsat;
    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e16m2(m - jj);

      vcols = __riscv_vlsseg4e16_v_u16m2x4(in_tile, input_seg_bstride, vl);

      uint16_t *write_ptr = out_tile;
      const size_t output_seg_stride = rsat;
      // store each segment continuously along m
      __riscv_vse16_v_u16m2(write_ptr, __riscv_vget_v_u16m2x4_u16m2(vcols, 0),
                            vl);
      write_ptr += output_seg_stride;
      __riscv_vse16_v_u16m2(write_ptr, __riscv_vget_v_u16m2x4_u16m2(vcols, 1),
                            vl);
      write_ptr += output_seg_stride;
      __riscv_vse16_v_u16m2(write_ptr, __riscv_vget_v_u16m2x4_u16m2(vcols, 2),
                            vl);
      write_ptr += output_seg_stride;
      __riscv_vse16_v_u16m2(write_ptr, __riscv_vget_v_u16m2x4_u16m2(vcols, 3),
                            vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using strided segment-2 loads and unit-strided
 * stores. Vectorization is performed along the M dimension, and N is expected
 * to be a multiple of 2.
 */
void
pacc_transpose_mvec_x2_e16_zve32x(size_t m, size_t n,
                                 const uint16_t *GGML_RESTRICT a, size_t rsa,
                                 uint16_t *GGML_RESTRICT at, size_t rsat) {

  const ptrdiff_t input_seg_bstride = (ptrdiff_t)(rsa * sizeof(uint16_t));

  size_t vl = 0;

  for (size_t ii = 0; ii + 1 < n; ii += 2) {

    // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
    vuint16m4x2_t vcols = __riscv_vundefined_u16m4x2();
    // NOLINTEND(clang-analyzer-deadcode.DeadStores)
    const uint16_t *in_tile = a + ii;
    uint16_t *out_tile = at + ii * rsat;
    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e16m4(m - jj);

      vcols = __riscv_vlsseg2e16_v_u16m4x2(in_tile, input_seg_bstride, vl);

      uint16_t *write_ptr = out_tile;
      const size_t output_seg_stride = rsat;
      // store each segment continuously along m
      __riscv_vse16_v_u16m4(write_ptr, __riscv_vget_v_u16m4x2_u16m4(vcols, 0),
                            vl);
      write_ptr += output_seg_stride;
      __riscv_vse16_v_u16m4(write_ptr, __riscv_vget_v_u16m4x2_u16m4(vcols, 1),
                            vl);

      in_tile += vl * rsa;
      out_tile += vl;
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using strided loads and unit-strided stores.
 * Vectorization is performed along the M dimension.
 */
void
pacc_transpose_mvec_x1_e16_zve32x(size_t m, size_t n,
                                 const uint16_t *GGML_RESTRICT a, size_t rsa,
                                 uint16_t *GGML_RESTRICT at, size_t rsat) {

  const ptrdiff_t input_bstride = (ptrdiff_t)(rsa * sizeof(uint16_t));

  size_t vl = 0;

  for (size_t ii = 0; ii < n; ii += 1) {
    const uint16_t *in_tile = a + ii;
    uint16_t *out_tile = at + ii * rsat;
    for (size_t jj = 0; jj < m; jj += vl) {
      vl = __riscv_vsetvl_e16m8(m - jj);
      vuint16m8_t v_data = __riscv_vlse16_v_u16m8(in_tile, input_bstride, vl);
      __riscv_vse16_v_u16m8(out_tile, v_data, vl);
      in_tile += vl * rsa;
      out_tile += vl;
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) with vectorization along the M dimension.
 */
void
pacc_transpose_mvec_e16_zve32x(size_t m, size_t n,
                              const uint16_t *GGML_RESTRICT a, size_t rsa,
                              uint16_t *GGML_RESTRICT at, size_t rsat) {

  size_t n_begin = 0;
  size_t n_end = n_begin + ((n - n_begin) / 8) * 8;

  pacc_transpose_mvec_x8_e16_zve32x(m, n_end - n_begin, a + n_begin, rsa,
                                   at + n_begin * rsat, rsat);

  n_begin = n_end;
  n_end = n_begin + ((n - n_begin) / 4) * 4;

  pacc_transpose_mvec_x4_e16_zve32x(m, n_end - n_begin, a + n_begin, rsa,
                                   at + n_begin * rsat, rsat);

  n_begin = n_end;
  n_end = n_begin + ((n - n_begin) / 2) * 2;

  pacc_transpose_mvec_x2_e16_zve32x(m, n_end - n_begin, a + n_begin, rsa,
                                   at + n_begin * rsat, rsat);

  n_begin = n_end;
  n_end = n;

  pacc_transpose_mvec_x1_e16_zve32x(m, n_end - n_begin, a + n_begin, rsa,
                                   at + n_begin * rsat, rsat);
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using unit-strided loads and strided segment-8
 * stores. Vectorization is performed along the N dimension, and M is expected
 * to be a multiple of 8.
 */
void
pacc_transpose_nvec_x8_e16_zve32x(size_t m, size_t n,
                                 const uint16_t *GGML_RESTRICT a, size_t rsa,
                                 uint16_t *GGML_RESTRICT at, size_t rsat) {
  size_t vl = 0;
  const ptrdiff_t output_seg_bstride = (ptrdiff_t)(rsat * sizeof(uint16_t));

  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  vuint16m1_t vrow0 = __riscv_vundefined_u16m1();
  vuint16m1_t vrow1 = __riscv_vundefined_u16m1();
  vuint16m1_t vrow2 = __riscv_vundefined_u16m1();
  vuint16m1_t vrow3 = __riscv_vundefined_u16m1();
  vuint16m1_t vrow4 = __riscv_vundefined_u16m1();
  vuint16m1_t vrow5 = __riscv_vundefined_u16m1();
  vuint16m1_t vrow6 = __riscv_vundefined_u16m1();
  vuint16m1_t vrow7 = __riscv_vundefined_u16m1();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)

  for (size_t ii = 0; ii + 7 < m; ii += 8) {
    for (size_t jj = 0; jj < n; jj += vl) {
      const uint16_t *read_ptr = a + ii * rsa + jj;
      vl = __riscv_vsetvl_e16m1(n - jj);

      vrow0 = __riscv_vle16_v_u16m1(read_ptr, vl);
      read_ptr += rsa;
      vrow1 = __riscv_vle16_v_u16m1(read_ptr, vl);
      read_ptr += rsa;
      vrow2 = __riscv_vle16_v_u16m1(read_ptr, vl);
      read_ptr += rsa;
      vrow3 = __riscv_vle16_v_u16m1(read_ptr, vl);
      read_ptr += rsa;
      vrow4 = __riscv_vle16_v_u16m1(read_ptr, vl);
      read_ptr += rsa;
      vrow5 = __riscv_vle16_v_u16m1(read_ptr, vl);
      read_ptr += rsa;
      vrow6 = __riscv_vle16_v_u16m1(read_ptr, vl);
      read_ptr += rsa;
      vrow7 = __riscv_vle16_v_u16m1(read_ptr, vl);

      vuint16m1x8_t vrows = __riscv_vcreate_v_u16m1x8(
          vrow0, vrow1, vrow2, vrow3, vrow4, vrow5, vrow6, vrow7);

      __riscv_vssseg8e16_v_u16m1x8(at + (jj * rsat) + ii, output_seg_bstride,
                                   vrows, vl);
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using unit-strided loads and strided segment-4
 * stores. Vectorization is performed along the N dimension, and M is expected
 * to be a multiple of 4.
 */
void
pacc_transpose_nvec_x4_e16_zve32x(size_t m, size_t n,
                                 const uint16_t *GGML_RESTRICT a, size_t rsa,
                                 uint16_t *GGML_RESTRICT at, size_t rsat) {
  size_t vl = 0;
  const ptrdiff_t output_seg_bstride = (ptrdiff_t)(rsat * sizeof(uint16_t));

  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  vuint16m2_t vrow0 = __riscv_vundefined_u16m2();
  vuint16m2_t vrow1 = __riscv_vundefined_u16m2();
  vuint16m2_t vrow2 = __riscv_vundefined_u16m2();
  vuint16m2_t vrow3 = __riscv_vundefined_u16m2();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)

  for (size_t ii = 0; ii + 3 < m; ii += 4) {
    for (size_t jj = 0; jj < n; jj += vl) {
      const uint16_t *read_ptr = a + ii * rsa + jj;
      vl = __riscv_vsetvl_e16m2(n - jj);
      vrow0 = __riscv_vle16_v_u16m2(read_ptr, vl);
      read_ptr += rsa;
      vrow1 = __riscv_vle16_v_u16m2(read_ptr, vl);
      read_ptr += rsa;
      vrow2 = __riscv_vle16_v_u16m2(read_ptr, vl);
      read_ptr += rsa;
      vrow3 = __riscv_vle16_v_u16m2(read_ptr, vl);

      vuint16m2x4_t vrows =
          __riscv_vcreate_v_u16m2x4(vrow0, vrow1, vrow2, vrow3);
      __riscv_vssseg4e16_v_u16m2x4(at + (jj * rsat) + ii, output_seg_bstride,
                                   vrows, vl);
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using unit-strided loads and strided segment-2
 * stores. Vectorization is performed along the N dimension, and M is expected
 * to be a multiple of 2.
 */
void
pacc_transpose_nvec_x2_e16_zve32x(size_t m, size_t n,
                                 const uint16_t *GGML_RESTRICT a, size_t rsa,
                                 uint16_t *GGML_RESTRICT at, size_t rsat) {
  size_t vl = 0;
  const ptrdiff_t output_seg_bstride = (ptrdiff_t)(rsat * sizeof(uint16_t));

  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  vuint16m4_t vrow0 = __riscv_vundefined_u16m4();
  vuint16m4_t vrow1 = __riscv_vundefined_u16m4();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)

  for (size_t ii = 0; ii + 1 < m; ii += 2) {
    for (size_t jj = 0; jj < n; jj += vl) {
      const uint16_t *read_ptr = a + ii * rsa + jj;
      vl = __riscv_vsetvl_e16m4(n - jj);
      vrow0 = __riscv_vle16_v_u16m4(read_ptr, vl);
      read_ptr += rsa;
      vrow1 = __riscv_vle16_v_u16m4(read_ptr, vl);

      vuint16m4x2_t vrows = __riscv_vcreate_v_u16m4x2(vrow0, vrow1);
      __riscv_vssseg2e16_v_u16m4x2(at + (jj * rsat) + ii, output_seg_bstride,
                                   vrows, vl);
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) using unit-strided loads and strided stores.
 * Vectorization is performed along the N dimension.
 */
void
pacc_transpose_nvec_x1_e16_zve32x(size_t m, size_t n,
                                 const uint16_t *GGML_RESTRICT a, size_t rsa,
                                 uint16_t *GGML_RESTRICT at, size_t rsat) {
  size_t vl = 0;
  const ptrdiff_t output_bstride = (ptrdiff_t)(rsat * sizeof(uint16_t));

  // NOLINTBEGIN(clang-analyzer-deadcode.DeadStores)
  vuint16m8_t vrow0 = __riscv_vundefined_u16m8();
  // NOLINTEND(clang-analyzer-deadcode.DeadStores)

  for (size_t ii = 0; ii < m; ii++) {
    for (size_t jj = 0; jj < n; jj += vl) {
      vl = __riscv_vsetvl_e16m8(n - jj);
      vrow0 = __riscv_vle16_v_u16m8(a + (ii + 0) * rsa + jj, vl);
      __riscv_vsse16_v_u16m8(at + (jj * rsat) + ii, output_bstride, vrow0, vl);
    }
  }
}

/* Transposes an M x N row-major matrix (pointed by `a`) to an N x M row-major
 * matrix (pointed by `at`) with vectorization along the N dimension.
 */
void
pacc_transpose_nvec_e16_zve32x(size_t m, size_t n,
                              const uint16_t *GGML_RESTRICT a, size_t rsa,
                              uint16_t *GGML_RESTRICT at, size_t rsat) {

  size_t m_begin = 0;
  size_t m_end = m_begin + ((m - m_begin) / 8) * 8;

  pacc_transpose_nvec_x8_e16_zve32x(m_end - m_begin, n, a + m_begin * rsa, rsa,
                                   at + m_begin, rsat);

  m_begin = m_end;
  m_end = m_begin + ((m - m_begin) / 4) * 4;

  pacc_transpose_nvec_x4_e16_zve32x(m_end - m_begin, n, a + m_begin * rsa, rsa,
                                   at + m_begin, rsat);

  m_begin = m_end;
  m_end = m_begin + ((m - m_begin) / 2) * 2;

  pacc_transpose_nvec_x2_e16_zve32x(m_end - m_begin, n, a + m_begin * rsa, rsa,
                                   at + m_begin, rsat);

  m_begin = m_end;
  m_end = m;

  pacc_transpose_nvec_x1_e16_zve32x(m_end - m_begin, n, a + m_begin * rsa, rsa,
                                   at + m_begin, rsat);
}

void pacc_transpose_e16_zve32x(size_t m, size_t n,
                                       const uint16_t *GGML_RESTRICT a,
                                       size_t rsa, uint16_t *GGML_RESTRICT at,
                                       size_t rsat) {
  if (pacc_transpose_e16_is_mvec(m, n)) {
    pacc_transpose_mvec_e16_zve32x(m, n, a, rsa, at, rsat);
  } else {
    pacc_transpose_nvec_e16_zve32x(m, n, a, rsa, at, rsat);
  }
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
	int32_t i1;
	int32_t i2;
};

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

	void * ptr = *p;
	ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
	*p = (void *) ((char *) ptr + size);
	return ptr;
}

template <int layout_block_n>
static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;
    ggml_vec_dot_t    const vec_dot      = ggml_get_type_traits_cpu(type)->vec_dot;
    enum ggml_type const vec_dot_type    = ggml_get_type_traits_cpu(type)->vec_dot_type;

    bool is_fp16_type = vec_dot_type == GGML_TYPE_F16;
    bool is_bf16_type = vec_dot_type == GGML_TYPE_BF16;
    bool is_q8_0_type = vec_dot_type == GGML_TYPE_Q8_0;
    assert(is_fp16_type || is_bf16_type || is_q8_0_type);

    const int64_t _i12 = ir1_start;  // logical row index for this expert

    struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
    const int               id          = row_mapping.i1;  // selected expert index

    const int64_t i11 = id % ne11;
    const int64_t i12 = row_mapping.i2;  // row index in src1

    const int64_t i1 = id;               // selected expert index
    const int64_t i2 = i12;              // row

    // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
    //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
    //       the original src1 data pointer, so we should index using the indices directly
    // TODO: this is a bit of a hack, we should probably have a better way to handle this
    const char * src1_col =
        (const char *) wdata +
        (src1_cont || src1->type != vec_dot_type ? (i11 + i12 * ne11) * row_size : (i11 * nb11 + i12 * nb12));

    float * dst_col = (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2));

    size_t tile_k_v = ne00;
    size_t tile_n_v = ir0_end - ir0_start;
    size_t tile_m_v = 1;

     if (is_bf16_type) {
            micro_kernel_bf16bf16fp32_tile_k1_tile_n_gemv<layout_block_n>(tile_m_v, tile_n_v, tile_k_v, (__bf16 *) src1_col,
                                                                  (__bf16 *) (src0_cur + ir0_start * nb01),
                                                                  &dst_col[ir0_start]);          
        } else if (is_fp16_type) {
            micro_kernel_fp16fp16fp32_tile_k1_tile_n_gemv<layout_block_n>(tile_m_v, tile_n_v, tile_k_v, (_Float16 *) src1_col,
                                                                  (_Float16 *) (src0_cur + ir0_start * nb01),
                                                                  &dst_col[ir0_start]);          
        } else {
            assert(0);
        }       
}

template <int layout_block_n>
static void ggml_compute_forward_mul_mat_one_chunk_mm(const struct ggml_compute_params * params,
                                                   struct ggml_tensor *               dst,
                                                   const enum ggml_type               type,
                                                   const int64_t                      num_rows_per_vec_dot,
                                                   const int64_t                      ir0_start,
                                                   const int64_t                      ir0_end,
                                                   const int64_t                      ir1_start,
                                                   const int64_t                      ir1_end) {
    //{type = GGML_TYPE_Q8_0, buffer = 0x4b62a90, ne = {1024, 6144, 1, 1}, nb = {34, 1088, 6684672, 6684672}, op = GGML_OP_NONE,
    const struct ggml_tensor * src0 = dst->src[0];

    //type = GGML_TYPE_F32, buffer = 0x91a180, ne = {1024, 4, 1, 1}, nb = {4, 4096, 16384, 16384}, op = GGML_OP_MUL,
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);
    const bool dst_cont  = ggml_is_contiguous(dst);

    int block_n = MIN((ir0_end - ir0_start), layout_block_n);

    assert(ir0_start % layout_block_n == 0);

    assert(src1_cont == true);
    assert(dst_cont == true);

    enum ggml_type const vec_dot_type = ggml_get_type_traits_cpu(type)->vec_dot_type;

    bool is_fp16_type = vec_dot_type == GGML_TYPE_F16;
    bool is_bf16_type = vec_dot_type == GGML_TYPE_BF16;
    bool is_q8_0_type = vec_dot_type == GGML_TYPE_Q8_0;
    assert(is_fp16_type || is_bf16_type || is_q8_0_type);

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;

    const char * new_src0 = (const char *) src0->data;                          //weight
    if (strncmp(src0->name, "token_embd.weight", 17) == 0)
      new_src0 += src0->nb[2];
      
      
    const int64_t M0 = ne11; // 4 in this case
    const int64_t N0 = ne01; // 2048 in this case
    const int64_t K0 = ne00; // 1024 in this case

    // 1tm4tn
    const uint16_t * lhs0_ptr = (const uint16_t *)wdata + ir1_start;
    const uint16_t * rhs0_ptr = (const uint16_t *)new_src0 + ir0_start * K0;

    const size_t dim_k = K0;

    size_t tm = 0;
    size_t tn = 0;
    size_t tk = 0;

    vuint16m4_t lhs0_data0 = __riscv_vundefined_u16m4();
    vuint16m4_t lhs0_data1 = __riscv_vundefined_u16m4();
    vuint16m4_t rhs0_data0 = __riscv_vundefined_u16m4();
    vuint16m4_t rhs0_data1 = __riscv_vundefined_u16m4();    

    __asm__ volatile("sf.vsettnt zero, zero, e16, w2");
    __asm__ volatile("sf.vsettm %0, %1" : "=r"(tm) : "r"(ir1_end - ir1_start));
    __asm__ volatile("sf.vsettn %0, %1" : "=r"(tn) : "r"(ir0_end - ir0_start));
    __asm__ volatile("sf.vtzero.t mt0");
    __asm__ volatile("sf.vtzero.t mt4");
    __asm__ volatile("sf.vtzero.t mt8");
    __asm__ volatile("sf.vtzero.t mt12");
    
    
    const int np = (block_n & ~(256 - 1));

    // We will use block_n which should be 256 in this case and 4 tile registers from 64T
    size_t n = 0;

    for (; n < np; n += 256) {
        __asm__ volatile("sf.vsettn %0, %1" : "=r"(tn) : "r"(block_n - n));

        __asm__ volatile("sf.vtzero.t mt0");
        __asm__ volatile("sf.vtzero.t mt4");
        __asm__ volatile("sf.vtzero.t mt8");
        __asm__ volatile("sf.vtzero.t mt12");

        lhs0_ptr = (const uint16_t *)wdata + ir1_start;
        const uint16_t * rhs0_block_ptr = rhs0_ptr + n;
        const uint16_t * rhs1_block_ptr = rhs0_ptr + n + 32;
        const uint16_t * rhs2_block_ptr = rhs0_ptr + n + 64;
        const uint16_t * rhs3_block_ptr = rhs0_ptr + n + 96;
        
        // For f16 data type, K_MAX is 2.
        __asm__ volatile("sf.vsettk %0, %1" : "=r"(tk) : "r"(dim_k));

        for (size_t k = 0; k < dim_k / 2; ++k) {
            // setup vl for tm dim
            __asm__ volatile("sf.vsettn zero, %0" : : "r"(tm));

            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(lhs0_data0) : "r"(lhs0_ptr));
            lhs0_ptr += M0;
            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(lhs0_data1) : "r"(lhs0_ptr));
            lhs0_ptr += M0;

            vuint16m8_t lhs0_data = __riscv_vcreate_v_u16m4_u16m8(lhs0_data0, lhs0_data1);

            // setup vl for tn dim
            __asm__ volatile("sf.vsettn zero, %0" : : "r"(tn));
            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(rhs0_data0) : "r"(rhs0_block_ptr));
            rhs0_block_ptr += block_n;
            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(rhs0_data1) : "r"(rhs0_block_ptr));
            rhs0_block_ptr += block_n;

            vuint16m8_t rhs0_data = __riscv_vcreate_v_u16m4_u16m8(rhs0_data0, rhs0_data1);

            __asm__ volatile("sf.mm.f.f mt0, %0, %1" : : "vr"(lhs0_data), "vr"(rhs0_data));

            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(rhs0_data0) : "r"(rhs1_block_ptr));
            rhs1_block_ptr += block_n;
            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(rhs0_data1) : "r"(rhs1_block_ptr));
            rhs1_block_ptr += block_n;
            rhs0_data = __riscv_vcreate_v_u16m4_u16m8(rhs0_data0, rhs0_data1);
            __asm__ volatile("sf.mm.f.f mt4, %0, %1" : : "vr"(lhs0_data), "vr"(rhs0_data));

            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(rhs0_data0) : "r"(rhs2_block_ptr));
            rhs2_block_ptr += block_n;
            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(rhs0_data1) : "r"(rhs2_block_ptr));
            rhs2_block_ptr += block_n;
            rhs0_data = __riscv_vcreate_v_u16m4_u16m8(rhs0_data0, rhs0_data1);
            __asm__ volatile("sf.mm.f.f mt8, %0, %1" : : "vr"(lhs0_data), "vr"(rhs0_data));
            
            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(rhs0_data0) : "r"(rhs3_block_ptr));
            rhs3_block_ptr += block_n;
            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(rhs0_data1) : "r"(rhs3_block_ptr));
            rhs3_block_ptr += block_n;
            rhs0_data = __riscv_vcreate_v_u16m4_u16m8(rhs0_data0, rhs0_data1);
            __asm__ volatile("sf.mm.f.f mt12, %0, %1" : : "vr"(lhs0_data), "vr"(rhs0_data));
        }

        size_t k_remainder = dim_k & 1;
        if (k_remainder) {
            __asm__ volatile("sf.vsettk %0, %1" : "=r"(tk) : "r"(k_remainder));
            // setup vl for tm dim
            __asm__ volatile("sf.vsettn zero, %0" : : "r"(tm));
            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(lhs0_data0) : "r"(lhs0_ptr));
            lhs0_ptr += M0;

            vuint16m8_t lhs0_data = __riscv_vcreate_v_u16m4_u16m8(lhs0_data0, lhs0_data1);

            // setup vl for tn dim
            __asm__ volatile("sf.vsettn zero, %0" : : "r"(tn));
            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(rhs0_data0) : "r"(rhs0_block_ptr));
            rhs0_block_ptr += block_n;

            vuint16m8_t rhs0_data = __riscv_vcreate_v_u16m4_u16m8(rhs0_data0, rhs0_data1);

            __asm__ volatile("sf.mm.f.f mt0, %0, %1" : : "vr"(lhs0_data), "vr"(rhs0_data));

            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(rhs0_data0) : "r"(rhs1_block_ptr));
            rhs1_block_ptr += block_n;
            rhs0_data = __riscv_vcreate_v_u16m4_u16m8(rhs0_data0, rhs0_data1);
            __asm__ volatile("sf.mm.f.f mt4, %0, %1" : : "vr"(lhs0_data), "vr"(rhs0_data));

            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(rhs0_data0) : "r"(rhs2_block_ptr));
            rhs2_block_ptr += block_n;
            rhs0_data = __riscv_vcreate_v_u16m4_u16m8(rhs0_data0, rhs0_data1);
            __asm__ volatile("sf.mm.f.f mt8, %0, %1" : : "vr"(lhs0_data), "vr"(rhs0_data));

            __asm__ volatile("vle16.v %0, (%1)" : "=vr"(rhs0_data0) : "r"(rhs3_block_ptr));
            rhs3_block_ptr += block_n;
            rhs0_data = __riscv_vcreate_v_u16m4_u16m8(rhs0_data0, rhs0_data1);
            __asm__ volatile("sf.mm.f.f mt12, %0, %1" : : "vr"(lhs0_data), "vr"(rhs0_data));
        }

        float * out00_ptr =
            (float *) ((char *) dst->data + (ir1_start * dst->nb[1]) + ((ir0_start + n) * sizeof(float)));  //result

        float * out01_ptr = (float *) ((char *) dst->data + (ir1_start * dst->nb[1]) +
                                       ((ir0_start + n + 32) * sizeof(float)));  //result
        float * out02_ptr = (float *) ((char *) dst->data + (ir1_start * dst->nb[1]) +
                                       ((ir0_start + n + 64) * sizeof(float)));  //result
        float * out03_ptr = (float *) ((char *) dst->data + (ir1_start * dst->nb[1]) +
                                       ((ir0_start + n + 96) * sizeof(float)));  //result

        const size_t tile_specifier_shift = 27;
        size_t       mt0_tss              = 0;
        size_t       mt4_tss              = 4 << tile_specifier_shift;
        size_t       mt8_tss              = 8 << tile_specifier_shift;
        size_t       mt12_tss             = 12 << tile_specifier_shift;

        for (size_t i = 0; i < tm; ++i) {
            __asm__ volatile("sf.vste32 %0, (%1)" : : "r"(mt0_tss++), "r"(out00_ptr));
            __asm__ volatile("sf.vste32 %0, (%1)" : : "r"(mt4_tss++), "r"(out01_ptr));
            __asm__ volatile("sf.vste32 %0, (%1)" : : "r"(mt8_tss++), "r"(out02_ptr));
            __asm__ volatile("sf.vste32 %0, (%1)" : : "r"(mt12_tss++), "r"(out03_ptr));
            out00_ptr += N0;
            out01_ptr += N0;
            out02_ptr += N0;
            out03_ptr += N0;
        }
    }

    assert(n != block_n);
}

class pacc_ext_tensor_traits : public tensor_traits_base {
  public:
    // just direct copy from
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        // not realy a GGML_TYPE_Q8_0 but same size.
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                {
                    size = ggml_row_size(GGML_TYPE_F16, ggml_nelements(op->src[1]));
                    return true;
                }
            case GGML_OP_MUL_MAT_ID:
                {
                    size = ggml_row_size(GGML_TYPE_F16, ggml_nelements(op->src[1])) + sizeof(int64_t);

                    const int64_t              n_as = op->src[0]->ne[2];  // n_as, n_expert
                    const int64_t              ne12 = op->src[1]->ne[2];  // n_tokens
                    const struct ggml_tensor * ids  = op->src[2];
                    // matrix_row_counts
                    size += n_as * sizeof(int64_t) + sizeof(int64_t);
                    // matrix_rows
                    size += n_as * ids->ne[0] * ids->ne[1] * sizeof(struct mmid_row_mapping) + sizeof(int64_t);

                    // atomic_current_chunk
                    size += CACHE_LINE_SIZE * n_as + CACHE_LINE_SIZE;

                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) {
                    forward_mul_mat_f16(params, op);
                    return true;
                }
            case GGML_OP_MUL_MAT_ID:
                if (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) {
                    forward_mul_mat_id_f16(params, op);
                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    void forward_mul_mat_id_f16(ggml_compute_params * params, ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * ids = op->src[2];
        ggml_tensor *              dst  = op;

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);


    enum ggml_type const vec_dot_type    = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
    const ggml_from_float_t from_float   = ggml_get_type_traits_cpu(src0->type)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = (void*)params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        (int64_t *)incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        (struct mmid_row_mapping *)incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        (char (*)[64])incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = (char *)params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, (int32_t)iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        //atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        //*current_chunk_ctr = nth;

        int *current_chunk_ctr = reinterpret_cast<int *>(atomic_current_chunk + cur_a);
        __atomic_store_n(current_chunk_ctr, nth, __ATOMIC_RELAXED);
    }

    ggml_barrier(params->threadpool);

    const int block_n = 256;

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata    = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size_n = block_n;
        int chunk_size_m = 1;

        int64_t nchunk0 = (nr0 + chunk_size_n - 1) / chunk_size_n;
        int64_t nchunk1 = (nr1 + chunk_size_m - 1) / chunk_size_m;


        const int64_t dr0 = chunk_size_n; 
        const int64_t dr1 = chunk_size_m;

        int current_chunk = ith;

        int *current_chunk_ctr = reinterpret_cast<int *>(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk<block_n>(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }
            current_chunk = __atomic_fetch_add(current_chunk_ctr, 1, __ATOMIC_RELAXED);
        }
    }
    }

    void from_float_with_transpose(const float * x,
                                   ggml_fp16_t * y,
                                   int64_t       row,
                                   int64_t       row_len,
                                   int64_t       col,
                                   int64_t       n) {
        int64_t i = 0;
#if defined(__riscv_zvfh)
        ptrdiff_t stride = row_len * 2;
        for (int vl; i < n; i += vl) {
            vl               = __riscv_vsetvl_e32m2(n - i);
            vfloat32m2_t  vx = __riscv_vle32_v_f32m2(&x[i], vl);
            vfloat16m1_t  vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
            ggml_fp16_t * np = y + (col + i) * row_len + row;
            __riscv_vsse16_v_f16m1((_Float16 *) np, stride, vy, vl);
            //__riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
        }
#endif
        for (; i < n; ++i) {
            //y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
            // y is the fixed, the wdata
            // col is the ne10_block_start
            // row_len is the total element number of one row in transposed matrix, should be ne01
            // row is the i11 from ne11, the original row number
            ggml_fp16_t * np = y + (col + i) * row_len + row;
            *np              = GGML_CPU_FP32_TO_FP16(x[i]);
        }
    }

    void forward_mul_mat_f16(ggml_compute_params * params, ggml_tensor * op) {
        const struct ggml_tensor * src0 = op->src[0];
        const struct ggml_tensor * src1 = op->src[1];
        ggml_tensor *              dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = params->ith;
        const int nth = params->nth;

        enum ggml_type const vec_dot_type    = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
        const ggml_from_float_t from_float   = ggml_get_type_traits_cpu(src0->type)->from_float;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_size(src0->type));
        GGML_ASSERT(nb10 == ggml_type_size(src1->type));

        GGML_ASSERT(nb11 == (ggml_type_size(src1->type) * ne10));
        GGML_ASSERT(nb12 == (nb11 * ne11));
        GGML_ASSERT(ne12 == 1);
        GGML_ASSERT(ne13 == 1);

        //GGML_ASSERT(nb00 == (ggml_type_size(src0->type) * ne00));
        //GGML_ASSERT(nb01 == (nb01 * ne01));
        GGML_ASSERT(ne02 == 1);
        GGML_ASSERT(ne03 == 1);
        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == sizeof(float));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        bool is_fp16_type = vec_dot_type == GGML_TYPE_F16;
        bool is_bf16_type = vec_dot_type == GGML_TYPE_BF16;
        bool is_q8_0_type = vec_dot_type == GGML_TYPE_Q8_0;
        //bool is_repack = (src0.extra != nullptr) & (src0.op == GGML_OP_NONE);
        GGML_ASSERT(is_fp16_type | is_bf16_type | is_q8_0_type);
        //GGML_ASSERT(is_repack);

        if (src1->type != vec_dot_type) {
            char * wdata = (char *) params->wdata;

            const size_t nbw0 = ggml_type_size(vec_dot_type);
            const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
            const size_t nbw2 = nbw1 * ne11;
            const size_t nbw3 = nbw2 * ne12;

            assert(params->wsize >= ne13 * nbw3);
            for (int64_t i13 = 0; i13 < ne13; ++i13) {
                for (int64_t i12 = 0; i12 < ne12; ++i12) {
                    for (int64_t i11 = 0; i11 < ne11; ++i11) {
                        size_t  bs               = ggml_blck_size(vec_dot_type);
                        int64_t ne10_block_start = (ith * ne10 / bs) / nth;
                        int64_t ne10_block_end   = ((ith + 1) * ne10 / bs) / nth;

                        from_float_with_transpose((float *) ((char *) src1->data + i13 * nb13 + i12 * nb12 +
                                                             i11 * nb11 + ne10_block_start * bs * nb10),
                                                  (ggml_fp16_t *) (wdata), i11, ne11, ne10_block_start,
                                                  (ne10_block_end - ne10_block_start) * bs);
                    }
                }
            }
        }

        if (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            //atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        const int block_n = 256;

        // ============================================================================
        // Block-N 对齐的分块策略
        // ============================================================================
        //
        // 目标：确保每个线程处理的数据量是 block_n 的倍数
        // 原因：硬件加速器（如 NFMA）要求数据对齐
        //       - BF16/FP16: block_n = 32  (NFMA = 8)
        //       - FP8:       block_n = 32 (NFMA = 16)
        //       - FP4:       block_n = 32 (NFMA = 32)
        //
        // 策略：
        //   1. chunk_size 向上对齐到 block_n 的倍数
        //   2. 每个线程的工作量向下对齐到 block_n 的倍数
        //   3. 不满足对齐要求的任务分配给空闲线程或跳过

        // This is the size of the first dimension of the result
        const int64_t nr0 = ne0;

        // This is the size of the rest of the dimensions of the result
        const int64_t nr1 = ne1 * ne2 * ne3;

        int     chunk_size_n = block_n;
        int     chunk_size_m = 64;

        int64_t nchunk0      = (nr0 + chunk_size_n - 1) / chunk_size_n;
        int64_t nchunk1      = (nr1 + chunk_size_m - 1) / chunk_size_m;

        // The number of elements in each chunk
        const int64_t dr0 = chunk_size_n;
        const int64_t dr1 = chunk_size_m;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        int current_chunk = ith;

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end   = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end   = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_one_chunk_mm<block_n>(params, dst, src0->type, 0, ir0_start, ir0_end,
                                                               ir1_start, ir1_end);

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) __riscv_vsetvlmax_e16m4(), (int) 1);

        int K = t->ne[0];
        int N = t->ne[1];
        int T = t->ne[2];

        uint16_t * dst_p = (uint16_t *) t->data;

        const int block_n = __riscv_vsetvlmax_e16m4();

        if (strncmp(t->name, "token_embd.weight", 17) == 0) {
            memcpy((char *) dst_p, (char *) data, t->nb[2]);
            dst_p += t->nb[2] / 2;
        }

        for (int t = 0; t < T; t++) {
            const uint16_t * cur_mat = (const uint16_t *) (data) + (t * N * K);
            const uint16_t * cur_row = cur_mat;

            int n = 0;

            for (; n + block_n - 1 < N; n += block_n) {
                pacc_transpose_mvec_e16_zve32x(block_n, K, cur_row, K, dst_p, block_n);
                cur_row += block_n * K;
                dst_p += block_n * K;
            }

            // Handle the tailing
            pacc_transpose_nvec_e16_zve32x((N - n), K, cur_row, K, dst_p, (N - n));
        }
        return 0;
    }
};

static const pacc_ext_tensor_traits pacc_tensor_traits;

}  // namespace ggml::cpu::riscv64_pacc

static const ggml::cpu::tensor_traits * ggml_riscv64_pacc_get_optimal_repack_type(const struct ggml_tensor * cur) {
    if (cur->type == GGML_TYPE_F16 || cur->type == GGML_TYPE_BF16) {
        if (cur->ne[1] % 16 == 0) {
            return &ggml::cpu::riscv64_pacc::pacc_tensor_traits;
        }
    }

    return nullptr;
}

static enum ggml_status ggml_backend_riscv64_pacc_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                     struct ggml_tensor *  tensor) {
    tensor->extra = (void *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_pacc_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_riscv64_pacc_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                        struct ggml_tensor *  tensor,
                                                        const void *          data,
                                                        size_t                offset,
                                                        size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_pacc::tensor_traits_base *) tensor->extra;
    if (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_riscv64_pacc_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_RISCV64_PACC";

    GGML_UNUSED(buft);
}

namespace ggml::cpu::riscv64_pacc {

class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_pacc_buffer_type() &&
                    ggml_riscv64_pacc_get_optimal_repack_type(op->src[0])) {
                    if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        return false;
                    }
                    if (op->src[1]->type == GGML_TYPE_F32) {
                        return true;
                    }
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_pacc_buffer_type() &&
                    ggml_riscv64_pacc_get_optimal_repack_type(op->src[0])) {
                    if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        return false;
                    }
                    if (op->src[1]->type == GGML_TYPE_F32) {
                        return true;
                    }
                }
                break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
                if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_pacc_buffer_type()) {
                    return (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
                //return (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_pacc::rvv_impl);
            default:
                // GGML_ABORT("fatal error");
                break;
        }

        return nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_pacc

static ggml_backend_buffer_t ggml_backend_cpu_riscv64_pacc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                    size_t                     size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_riscv64_pacc_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_riscv64_pacc_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cpu_riscv64_pacc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cpu_riscv64_pacc_nbytes(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }

    size_t       nbytes;
    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
        }
    } else {
        nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;
        if (tensor->type == GGML_TYPE_Q4_K) {
            GGML_ASSERT(nbytes % sizeof(block_q4_K) == 0);
            nbytes = (nbytes / sizeof(block_q4_K)) * sizeof(block_q4_1) * 8;
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                nbytes += (tensor->ne[i] - 1) * (tensor->nb[i] / sizeof(block_q4_K)) * sizeof(block_q4_1) * 8;
            }
        } else {
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
            }
        }
    }

    if (strncmp(tensor->name, "token_embd.weight", 17) == 0) {
	    nbytes = 2 * nbytes + sizeof(int64_t);
    }

    GGML_UNUSED(buft);
    return nbytes;
}

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_pacc_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_pacc = {
        /* .iface    = */
        {
         /* .get_name         = */ ggml_backend_cpu_riscv64_pacc_buffer_type_get_name,
         /* .alloc_buffer     = */ ggml_backend_cpu_riscv64_pacc_buffer_type_alloc_buffer,
         /* .get_alignment    = */ ggml_backend_cpu_riscv64_pacc_buffer_type_get_alignment,
         /* .get_max_size     = */ nullptr,
         /* .get_alloc_size   = */ ggml_backend_cpu_riscv64_pacc_nbytes,
         /* .is_host          = */ nullptr,
         },
        /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */
        new ggml::cpu::riscv64_pacc::extra_buffer_type(),
    };

    return &ggml_backend_cpu_buffer_type_riscv64_pacc;
}
