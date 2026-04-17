#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "pacc_v0.h"

#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "traits.h"

#include "pacc_matmul.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>  // for GGML_ASSERT
#include <stdexcept>
#include <thread>

#include <utility>
#include <riscv_vector.h>

#if defined(__GNUC__)
// #pragma GCC diagnostic ignored "-Woverlength-strings"
// #pragma GCC diagnostic ignored "-Wcast-qual"
// #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

namespace ggml::cpu::riscv64_pacc_v0 {

}  // namespace ggml::cpu::riscv64_pacc_v0

namespace ggml::cpu::riscv64_pacc_v0 {

class tensor_traits_base : public ggml::cpu::tensor_traits {
  public:
    virtual int repack(struct ggml_tensor * t, const void * data, size_t data_size) = 0;
};

class tensor_traits_common : public tensor_traits_base {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        switch (op->op) {
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
                size = 0;
                return true;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }


    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        struct PACC_fd * pacc_fd = pacc_v0_get_fds();
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) {
                    if (params->ith < pacc_fd->count) {
                        printf("ith: %d, pacc_fd: %d\n", params->ith, pacc_fd->pacc_device_fds[params->ith]);
                        pacc_v0_compute_forward_mul_mat(params, op);
                    }
                    return true;
                }
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    enum LaunchKernelType {
        LOCAL_TEST_FP16,
        LOCAL_TEST_BF16,
        PACC_FP16,
        PACC_BF16,
    };

    static void my_vec_dot_f16(int n, float * __restrict__ s, const uint16_t * __restrict__ x, const uint16_t * __restrict__ y) {
        float sumf = 0.0;

        int vl = __riscv_vsetvlmax_e32m2();
        vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m2_t vsum;
        vfloat16m1_t ax;
        vfloat16m1_t ay;
        vsum = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vmv_v_x_u32m2(0, vl));
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m1(n - i);
            ax = __riscv_vle16_v_f16m1_tu(ax, (const _Float16 *)&x[i], vl);
            ay = __riscv_vle16_v_f16m1_tu(ay, (const _Float16 *)&y[i], vl);
            vsum = __riscv_vfwmacc_vv_f32m2_tu(vsum, ax, ay, vl);
        }
        vl = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t ac0 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(vsum, 0), __riscv_vget_v_f32m2_f32m1(vsum, 1), vl);
        vs = __riscv_vfredusum_vs_f32m1_f32m1(ac0, vs, vl);
        sumf += __riscv_vfmv_f_s_f32m1_f32(vs);

        *s = sumf;
    }

    static void my_vec_dot_bf16(int n, float * __restrict__ s, const uint16_t * __restrict__ x, const uint16_t * __restrict__ y) {
        float sumf = 0.0;

        int vl = __riscv_vsetvlmax_e32m2();
        vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m2_t vsum;
        vbfloat16m1_t ax;
        vbfloat16m1_t ay;
        vsum = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vmv_v_x_u32m2(0, vl));
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m1(n - i);
            ax = __riscv_vle16_v_bf16m1_tu(ax, (const __bf16 *)&x[i], vl);
            ay = __riscv_vle16_v_bf16m1_tu(ay, (const __bf16 *)&y[i], vl);
            vsum = __riscv_vfwmaccbf16_vv_f32m2_tu(vsum, ax, ay, vl);
        }
        vl = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t ac0 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(vsum, 0), __riscv_vget_v_f32m2_f32m1(vsum, 1), vl);
        vs = __riscv_vfredusum_vs_f32m1_f32m1(ac0, vs, vl);
        sumf += __riscv_vfmv_f_s_f32m1_f32(vs);

        *s = sumf;
    }

    void localKernelFP16(int m, int n, int k, const uint16_t *A, int lda, const uint16_t *B, int ldb, float *C, int ldc) {
        for (int mm = 0; mm < m; ++mm) {
            for (int nn = 0; nn < n; ++nn) {
                my_vec_dot_f16(k, &C[mm + nn * m], &A[mm * k], &B[nn * k]);
            }
        }
    }

    void localKernelBF16(int m, int n, int k, const uint16_t *A, int lda, const uint16_t *B, int ldb, float *C, int ldc) {
        for (int mm = 0; mm < m; ++mm) {
            for (int nn = 0; nn < n; ++nn) {
                my_vec_dot_bf16(k, &C[mm + nn * m], &A[mm * k], &B[nn * k]);
            }
        }
    }

    void pacc_mat_mul_wrapper(int m, int n, int k, const uint16_t *A, int lda, const uint16_t *B, int ldb, float *C, int ldc, int pacc_fd, enum LaunchKernelType t) {
        if (t == LOCAL_TEST_FP16 || t == LOCAL_TEST_BF16) {
            if (t == LOCAL_TEST_FP16) {
                localKernelFP16(m, n, k, A, lda, B, ldb, C, ldc);
            } else if (t == LOCAL_TEST_BF16) {
                localKernelBF16(m, n, k, A, lda, B, ldb, C, ldc);
            }
        } else if (t == PACC_FP16 || t == PACC_BF16) {
            std::pair<void*, create_bo> bufA;
            int bufA_size = m * lda;
            share_memory_alloc(&bufA.first, bufA_size, pacc_fd, &bufA.second);
            std::pair<void*, create_bo> bufB;
            int bufB_size = n * ldb;
            share_memory_alloc(&bufB.first, bufB_size, pacc_fd, &bufB.second);
            std::pair<void*, create_bo> bufC;
            int bufC_size = m * ldc;
            share_memory_alloc(&bufC.first, bufC_size, pacc_fd, &bufC.second);

            memcpy(bufA.first, A, bufA_size);
            memcpy(bufB.first, B, bufB_size);

            pacc_error_t err;
            if (t == PACC_FP16) {
                err = pacc_mul_mat_f16(m, n, k, (const uint16_t *)bufA.first, lda, (const uint16_t *)bufB.first, ldb, (float *)bufC.first, ldc, pacc_fd, &bufA.second, &bufB.second, &bufC.second);
            } else if (t == PACC_BF16) {
                err = pacc_mul_mat_bf16(m, n, k, (const uint16_t *)bufA.first, lda, (const uint16_t *)bufB.first, ldb, (float *)bufC.first, ldc, pacc_fd, &bufA.second, &bufB.second, &bufC.second);
            }

            if (err != paccSuccess) {
                GGML_ABORT("call pacc function with error: %d. ", err);
            }

            memcpy(C, bufC.first, bufC_size);
        }
    }

    void pacc_v0_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
        const struct ggml_tensor * src0 = dst->src[0];
        const struct ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = params->ith;
        const int nth = params->nth;

        const int64_t r2 = ne12 / ne02;
        const int64_t r3 = ne13 / ne03;

        const bool src1_cont = ggml_is_contiguous(src1);
        pacc_error_t err;

        struct PACC_fd * pacc_fd = pacc_v0_get_fds();
        if (src1_cont) {

            int n = ggml_nelements(src1);
            void * work_data = nullptr;
            if (src0->type == GGML_TYPE_F16) {
                work_data = new ggml_fp16_t[n];
                ggml_fp32_to_fp16_row((float *)src1->data, (ggml_fp16_t *)work_data, n);
            } if (src0->type == GGML_TYPE_BF16) {
                work_data = new ggml_bf16_t[n];
                ggml_fp32_to_bf16_row((float *)src1->data, (ggml_bf16_t *)work_data, n);
            }

            for (int64_t i13 = 0; i13 < ne13; i13++) {
                for (int64_t i12 = 0; i12 < ne12; i12++) {
                    LaunchKernelType t = LOCAL_TEST_FP16;

                    if (src0->type == GGML_TYPE_F16) {
                        t = PACC_FP16;
                    } else if (src0->type == GGML_TYPE_BF16) {
                        t = PACC_BF16;
                    } else {
                        GGML_ABORT("unsupport datatype: %d. ", src0->type);
                    }

                    t = LOCAL_TEST_BF16;

                    pacc_mat_mul_wrapper(
                        ne01,
                        ne11,
                        ne00/ggml_blck_size(src0->type),
                        (const uint16_t *)((const char *)src0->data + i12/r2*nb02 + i13/r3*nb03),
                        nb01/ggml_type_size(src0->type) * ggml_type_size(src0->type),
                        (const uint16_t *)((const char *)work_data + i12*nb12 + i13*nb13),
                        nb11/ggml_type_size(src1->type) * ggml_type_size(src1->type),
                        (float *)((char *)dst->data + i12*nb2 + i13*nb3),
                        nb1/ggml_type_size(dst->type) * ggml_type_size(src1->type),
                        pacc_fd->pacc_device_fds[0],
                        t
                    );
                }
            }

            delete[] work_data;
        }

        ggml_barrier(params->threadpool);
    }


    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        memcpy(t->data, data, data_size);
        return 0;
    }
};

static const tensor_traits_common rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

static const ggml::cpu::tensor_traits * ggml_riscv64_pacc_v0_get_optimal_repack_type(const struct ggml_tensor * cur) {
    if (cur->type == GGML_TYPE_F16 || cur->type == GGML_TYPE_BF16) {
        // TODO add my traits
        return &ggml::cpu::riscv64_pacc_v0::rvv_impl;
    }

    return nullptr;
}

static enum ggml_status ggml_backend_riscv64_pacc_v0_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         struct ggml_tensor *  tensor) {
    tensor->extra =
        (void *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_pacc_v0_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_riscv64_pacc_v0_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            struct ggml_tensor *  tensor,
                                                            const void *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_pacc_v0::tensor_traits_base *) tensor->extra;
    if (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_riscv64_pacc_v0_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_RISCV64_PACC_V0";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_riscv64_pacc_v0_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_riscv64_pacc_v0_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_riscv64_pacc_v0_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cpu_riscv64_pacc_v0_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cpu_riscv64_pacc_v0_nbytes(ggml_backend_buffer_type_t buft,
                                                       const struct ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }
    return  ggml_nbytes(tensor);
}

namespace ggml::cpu::riscv64_pacc_v0 {

class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                return false;
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_pacc_v0_buffer_type()) {
                    return (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }

        return nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_paacc_v0

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_pacc_v0_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_pacc_v0 = {
  /* .iface    = */
        {
         /* .get_name         = */ ggml_backend_cpu_riscv64_pacc_v0_buffer_type_get_name,
         /* .alloc_buffer     = */ ggml_backend_cpu_riscv64_pacc_v0_buffer_type_alloc_buffer,
         /* .get_alignment    = */ ggml_backend_cpu_riscv64_pacc_v0_buffer_type_get_alignment,
         /* .get_max_size     = */ nullptr,
         /* .get_alloc_size   = */ ggml_backend_cpu_riscv64_pacc_v0_nbytes,
         /* .is_host          = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        new ggml::cpu::riscv64_pacc_v0::extra_buffer_type(),
    };

    return &ggml_backend_cpu_buffer_type_riscv64_pacc_v0;
}
