#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "pacc_v1.h"

#include "vec.h"
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "traits.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>  // for GGML_ASSERT
#include <stdexcept>
#include <thread>

#include <utility>

#if defined(__GNUC__)
// #pragma GCC diagnostic ignored "-Woverlength-strings"
// #pragma GCC diagnostic ignored "-Wcast-qual"
// #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

namespace ggml::cpu::riscv64_pacc_v1 {

}  // namespace ggml::cpu::riscv64_pacc_v1

namespace ggml::cpu::riscv64_pacc_v1 {

class tensor_traits_base : public ggml::cpu::tensor_traits {
  public:
    virtual int repack(struct ggml_tensor * t, const void * data, size_t data_size) = 0;
};

enum TensorType
{
    Local,
    Remote,
};

struct tensor_traits_common : public tensor_traits_base {
    TensorType t = Local;
    std::vector<void*> ptrs;

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

    bool is_remote(struct ggml_tensor * t) {
        auto weight = t->src[0];
        if (weight && weight->extra) {
            auto tensor_info = (ggml::cpu::riscv64_pacc_v1::tensor_traits_common *)weight->extra;
            return tensor_info->t == Remote;
        }
        return false;
    }

    void pacc_v1_compute_forward_mul_mat_remote(
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

        if (src1_cont) {

            int n = ggml_nelements(src1);
            void * work_data = nullptr;
            work_data = malloc(sizeof(uint16_t) * n);
            if (src0->type == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row((float *)src1->data, (ggml_fp16_t *)work_data, n);
            } if (src0->type == GGML_TYPE_BF16) {
                ggml_fp32_to_bf16_row((float *)src1->data, (ggml_bf16_t *)work_data, n);
            }

            for (int64_t i13 = 0; i13 < ne13; i13++) {
                for (int64_t i12 = 0; i12 < ne12; i12++) {
                    LaunchKernelType t = LOCAL_TEST_BF16;

                    auto tensor_info = (ggml::cpu::riscv64_pacc_v1::tensor_traits_common *)src0->extra;
                    const int slice_count = tensor_info->ptrs.size();

                    float * dst_part = new float[nb1 * ne1 / slice_count];

                    for (int i = 0; i < slice_count; ++i) {

                        if (src0->type == GGML_TYPE_F16) {
                            t = PACC_FP16;
                        } else if (src0->type == GGML_TYPE_BF16) {
                            t = PACC_BF16;
                        } else {
                            GGML_ABORT("unsupport datatype: %d. ", src0->type);
                        }

                        t = LOCAL_TEST_BF16;

                        const uint16_t * weight_parts = (const uint16_t *)((const char *)tensor_info->ptrs.at(i) + i12/r2*nb02 + i13/r3*nb03);
                        const int weight_size = ne01 / slice_count;

                        pacc_mat_mul_wrapper(
                            /* m */ weight_size,
                            /* n */ ne11,
                            /* k */ ne00,
                            /* A */ weight_parts,
                            /* lda */ nb01,
                            /* B */ (const uint16_t *)((const char *)work_data + i12*nb12 + i13*nb13),
                            /* ldb */ nb11,
                            /* C */ dst_part,
                            /* ldc */ nb1 / slice_count,
                            0,
                            t
                        );

                        /* i is slice num. */
                        for (int j = 0; j < ne1; ++j) {
                            memcpy((char *)dst->data + i12*nb2 * i13*nb3 + i * (nb1 / slice_count) + j * nb1, dst_part, nb1 / slice_count);
                        }
                    }

                    delete[] dst_part;

                }
            }

            free(work_data);
        }

        ggml_barrier(params->threadpool);
    }


    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:

                if (is_remote(op)) {
                    pacc_v1_compute_forward_mul_mat_remote(params, op);
                    return true;
                }

                if (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) {
                    if (params->ith < 1) {
                        /// printf("ith: %d, pacc_fd: %d\n", params->ith, pacc_fd->pacc_device_fds[params->ith]);
                        pacc_v1_compute_forward_mul_mat(params, op);
                    }
                }
                return false;
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

    void localKernelFP16(int m, int n, int k, const uint16_t *A, int lda, const uint16_t *B, int ldb, float *C, int ldc) {
        for (int mm = 0; mm < m; ++mm) {
            for (int nn = 0; nn < n; ++nn) {
                ggml_vec_dot_f16(k, &C[mm + nn * m], 0, (ggml_fp16_t *)&A[mm * k], 0, (ggml_fp16_t *)&B[nn * k], 0, 1);
            }
        }
    }

    void localKernelBF16(int m, int n, int k, const uint16_t *A, int lda, const uint16_t *B, int ldb, float *C, int ldc) {
        for (int mm = 0; mm < m; ++mm) {
            for (int nn = 0; nn < n; ++nn) {
                ggml_vec_dot_bf16(k, &C[mm + nn * m], 0, (ggml_bf16_t *)&A[mm * k], 0, (ggml_bf16_t *)&B[nn * k], 0, 1);
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
            GGML_ABORT("unreachable branch.\n");
        }
    }

    void pacc_v1_compute_forward_mul_mat(
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

        if (src1_cont) {

            int n = ggml_nelements(src1);
            void * work_data = nullptr;
            work_data = malloc(sizeof(uint16_t) * n);
            if (src0->type == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row((float *)src1->data, (ggml_fp16_t *)work_data, n);
            } if (src0->type == GGML_TYPE_BF16) {
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
                        0,
                        t
                    );
                }
            }

            free(work_data);
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

static ggml::cpu::riscv64_pacc_v1::tensor_traits_common * ggml_riscv64_pacc_v1_get_optimal_repack_type(const struct ggml_tensor * cur) {
    if (cur->type == GGML_TYPE_F16 || cur->type == GGML_TYPE_BF16) {
        // TODO add my traits
        // return &ggml::cpu::riscv64_pacc_v1::rvv_impl;
        return new ggml::cpu::riscv64_pacc_v1::tensor_traits_common;
    }

    return nullptr;
}

static bool is_slice_tensor(const char * name)
{
    const char * weight_suffix = "weight";
    const char * tok = "token_embd";

    const bool is_weight = (strstr(name, weight_suffix) != nullptr);
    const bool is_embd = (strstr(name, tok) != nullptr);

    return is_weight && !is_embd;
}
static const int slice_num = 4;

static enum ggml_status ggml_backend_riscv64_pacc_v1_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         struct ggml_tensor *  tensor) {
    auto * traits = const_cast<ggml::cpu::riscv64_pacc_v1::tensor_traits_common *>(ggml_riscv64_pacc_v1_get_optimal_repack_type(tensor));
    if (is_slice_tensor(ggml_get_name(tensor)) && ((ggml_nbytes(tensor) % 4) == 0)) {
        traits->t = ggml::cpu::riscv64_pacc_v1::Remote;
    } else {
        traits->t = ggml::cpu::riscv64_pacc_v1::Local;
    }
    tensor->extra = traits;

    GGML_UNUSED(buffer);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_riscv64_pacc_v1_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            struct ggml_tensor *  tensor,
                                                            const void *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto * tensor_info = (ggml::cpu::riscv64_pacc_v1::tensor_traits_common *) tensor->extra;
    if (tensor_info && tensor_info->t == ggml::cpu::riscv64_pacc_v1::Local) {
    // if (tensor_info) {
        auto OK = tensor_info->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    if (tensor_info && tensor_info->t == ggml::cpu::riscv64_pacc_v1::Remote) {
        const int chunk_size = size / slice_num;
        assert((size % slice_num) == 0);

        for (int i = 0; i < slice_num; ++i) {
            auto cur = new char[chunk_size];
            memcpy(cur, (char *)data + i * chunk_size, chunk_size);
            tensor_info->ptrs.push_back(cur);
        }
    }

    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_riscv64_pacc_v1_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_RISCV64_pacc_v1";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_riscv64_pacc_v1_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_riscv64_pacc_v1_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_riscv64_pacc_v1_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cpu_riscv64_pacc_v1_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cpu_riscv64_pacc_v1_nbytes(ggml_backend_buffer_type_t buft,
                                                       const struct ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }
    return  ggml_nbytes(tensor);
}

namespace ggml::cpu::riscv64_pacc_v1 {

class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        auto weight = op->src[0];
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                return is_slice_tensor(ggml_get_name(weight)) && ((ggml_nbytes(weight) % 4) == 0);
                break;
            default:
                // GGML_ABORT("fatal error");
                return false;
                break;
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->buffer) {
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

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_pacc_v1_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_pacc_v1 = {
  /* .iface    = */
        {
         /* .get_name         = */ ggml_backend_cpu_riscv64_pacc_v1_buffer_type_get_name,
         /* .alloc_buffer     = */ ggml_backend_cpu_riscv64_pacc_v1_buffer_type_alloc_buffer,
         /* .get_alignment    = */ ggml_backend_cpu_riscv64_pacc_v1_buffer_type_get_alignment,
         /* .get_max_size     = */ nullptr,
         /* .get_alloc_size   = */ ggml_backend_cpu_riscv64_pacc_v1_nbytes,
         /* .is_host          = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        new ggml::cpu::riscv64_pacc_v1::extra_buffer_type(),
    };

    return &ggml_backend_cpu_buffer_type_riscv64_pacc_v1;
}
