#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

#define MAX_NARGS 3

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define GGML_SILU_FP16

//
// logging
//

#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

#define GGML_PRINT(...) printf(__VA_ARGS__)

static float frand(void) {
    return (float)rand()/(float)RAND_MAX;
}

static int irand(int n) {
    if (n == 0) return 0;
    return rand()%n;
}

static void get_random_dims(int64_t * dims, int ndims) {
    dims[0] = dims[1] = dims[2] = dims[3] = 1;

    for (int i = 0; i < ndims; i++) {
        dims[i] = 1 + irand(4);
    }
}

static struct ggml_tensor * get_random_tensor_f32(
        struct ggml_context * ctx0,
        int ndims,
        const int64_t ne[],
        float fmin,
        float fmax) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);

    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)result->data)[i0] = frand()*(fmax - fmin) + fmin;
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)result->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)result->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)result->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    };

    return result;
}

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads, nullptr);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

static void show_matmul_tensor_fp32(int64_t m, int64_t n, int64_t k,
                                    const void *A, int64_t lda,
                                    const void *B, int64_t ldb,
                                    const void *C, int64_t ldc) {

    printf("A:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            const float *pos = (const float *)A + i * lda + j;

            printf("%f ", *pos);
        }
        printf("\n");
    }
    printf("B:\n");
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            const float *pos = (const float *)B + i + j * ldb;

            printf("%f ", *pos);
        }
        printf("\n");
    }
    printf("C:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const float *pos = (const float *)C + i + j * ldc;

            printf("%f ", *pos);
        }
        printf("\n");
    }
}

static void test_simple_matmul()
{
    struct ggml_init_params params = {
        /* .mem_size   = */ 128*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    std::vector<uint8_t> work_buffer;

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, 4);
    GGML_TENSOR_LOCALS(int32_t, a_ne, tensor_a, ne);
    GGML_TENSOR_LOCALS(size_t, a_nb, tensor_a, nb);

    float * tensor_a_data = (float *) tensor_a->data;
    for (int64_t i1 = 0; i1 < a_ne1; i1++) {
        for (int64_t i0 = 0; i0 < a_ne0; i0++) {
            int64_t idx = (i0 + i1 * a_ne0);
            if ((i1 + i0) % 2 == 0) {
                tensor_a_data[idx] = 0;
            } else {
                tensor_a_data[idx] = 1;
            }
        }
    }
    int n = ggml_nelements(tensor_a);
    for (int i = 0; i < n; ++i) {
        // tensor_a_data[i] = (i % 2);
    }

    struct ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, 3);
    GGML_TENSOR_LOCALS(int32_t, b_ne, tensor_b, ne);
    GGML_TENSOR_LOCALS(size_t, b_nb, tensor_b, nb);

    float * tensor_b_data = (float *) tensor_b->data;
    int v = 0;
    for (int64_t i1 = 0; i1 < b_ne1; i1++) {
        for (int64_t i0 = 0; i0 < b_ne0; i0++) {
            int64_t idx = (i0 + i1 * b_ne0);
            tensor_b_data[idx] = v++;
        }
    }

    int nn = ggml_nelements(tensor_a);
    for (int i = 0; i < nn; ++i) {
        tensor_b_data[i] = i;
    }

    struct ggml_tensor * r0 = ggml_mul_mat(ctx0, tensor_a, tensor_b);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    ggml_build_forward_expand(gf, r0);

    ggml_graph_compute_helper(work_buffer, gf, 1);

    const float * r0_data = (float *) r0->data;
    GGML_TENSOR_LOCALS(int32_t, r0_ne, r0, ne);

    printf("ne1: %d, ne0: %d\n", r0_ne1, r0_ne0);
    for (int64_t i0 = 0; i0 < r0_ne0; i0++) {
        for (int64_t i1 = 0; i1 < r0_ne1; i1++) {
            // int64_t idx = (i0 * b_nb0 + i1 * b_nb1);
            int64_t idx = (4 * i1 + i0);
            printf("%d ", (int)r0_data[idx]);
        }
        printf("\n");
    }

    ggml_free(ctx0);
}

int main(int /*argc*/, const char ** /*argv*/) {
    test_simple_matmul();

    return 0;
}
