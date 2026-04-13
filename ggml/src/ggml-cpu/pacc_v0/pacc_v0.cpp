#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "pacc_v0.h"

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
                if (params->ith < pacc_fd->count) {
                    printf("ith: %d, pacc_fd: %d\n", params->ith, pacc_fd->pacc_device_fds[params->ith]);
                    pacc_v0_compute_forward_mul_mat(params, op);
                }
                return true;
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
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
            for (int64_t i13 = 0; i13 < ne13; i13++) {
                for (int64_t i12 = 0; i12 < ne12; i12++) {
                    if (src1->type == GGML_TYPE_F16) {
                        err = pacc_mul_mat_f16(
                            ne01,
                            ne11,
                            ne00/ggml_blck_size(src0->type),
                            (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                            nb01/ggml_type_size(src0->type) * ggml_type_size(src0->type),
                            (const char *)src1->data + i12*nb12 + i13*nb13,
                            nb11/ggml_type_size(src1->type) * ggml_type_size(src1->type),
                            (char *)dst->data + i12*nb2 + i13*nb3,
                            nb1/ggml_type_size(dst->type) * ggml_type_size(src1->type),
                            0,
                            pacc_fd->pacc_device_fds[0],
                           );
                    } else if (src1->type == GGML_TYPE_BF16) {
                        err = pacc_mul_mat_bf16(
                            ne01,
                            ne11,
                            ne00/ggml_blck_size(src0->type),
                            (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                            nb01/ggml_type_size(src0->type) * ggml_type_size(src1->type),
                            (const char *)src1->data + i12*nb12 + i13*nb13,
                            nb11/ggml_type_size(src1->type) * ggml_type_size(src1->type),
                            (char *)dst->data + i12*nb2 + i13*nb3,
                            nb1/ggml_type_size(dst->type) * ggml_type_size(src1->type),
                            0,
                            pacc_fd->pacc_device_fds[0],
                           );
                    }

                    if (err != paccSuccess) {
                        GGML_ABORT("device mul mat error. ");
                    }
                }
            }

        }
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
                return true;
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
