#include "ggml.h"
#include "ggml-cpu.h"
#include <string.h>
#include <stdio.h>
#include <vector>

using namespace std;

void mul_mat_test_fp16(int m, int k, int n)
{
    // initialize data of matrices to perform matrix multiplication
    const int rows_A = m, cols_A = k;

    vector<float> matrix_A(rows_A * cols_A);
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_A; ++j) {
            matrix_A.at(i * cols_A + j) = j;
        }
    }

    const int rows_B = n, cols_B = k;

    vector<float> matrix_B(rows_B * cols_B);
    for (int i = 0; i < rows_B; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            matrix_B.at(i * cols_B + j) = j;
        }
    }

    // 1. Allocate `ggml_context` to store tensor data
    // Calculate the size needed to allocate
    size_t ctx_size = 0;
    ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F16); // tensor a
    ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F16); // tensor b
    ctx_size += rows_A * rows_B * ggml_type_size(GGML_TYPE_F32); // result
    ctx_size += 3 * ggml_tensor_overhead(); // metadata for 3 tensors
    ctx_size += ggml_graph_overhead(); // compute graph
    ctx_size += 1024; // some overhead (exact calculation omitted for simplicity)

    // Allocate `ggml_context` to store tensor data
    struct ggml_init_params params = {
        /*.mem_size =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 2. Create tensors and set data
    struct ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cols_A, rows_A);
    struct ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, cols_B, rows_B);

    ggml_fp32_to_fp16_row(matrix_A.data(), (ggml_fp16_t *)tensor_a->data, ggml_nelements(tensor_a));
    ggml_fp32_to_fp16_row(matrix_B.data(), (ggml_fp16_t *)tensor_b->data, ggml_nelements(tensor_b));

    // 3. Create a `ggml_cgraph` for mul_mat operation
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // result = a*b^T
    // Pay attention: ggml_mul_mat(A, B) ==> B will be transposed internally
    // the result is transposed
    struct ggml_tensor * result = ggml_mul_mat(ctx, tensor_a, tensor_b);

    // Mark the "result" tensor to be computed
    ggml_build_forward_expand(gf, result);

    // 4. Run the computation
    int n_threads = 1; // Optional: number of threads to perform some operations with multi-threading
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    // 5. Retrieve results (output tensors)
    float * result_data = (float *) result->data;
    printf("m:%d, n:%d, k:%d\n, mul mat (%d x %d) (transposed result):\n[", m, n, k, (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1]/* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0]/* cols */; i++) {
            printf(" %.2f", result_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");

    for (int i = 1; i < ggml_nelements(result); ++i) {
        if (result_data[i] != result_data[i - 1]) {
            printf("result error, m: %d, n: %d, k: %d\n", m, n, k);
            break;
        }
    }

    // 6. Free memory and exit
    ggml_free(ctx);
}

struct DimsInfo
{
    int m = 0;
    int n = 0;
    int k = 0;
    int counter = 0;

    DimsInfo(int mm, int nn, int kk, int cc)
    :m(mm), n(nn), k(kk), counter(cc) {}
};


int main(void) {
    int seq_len = 8;
    vector<DimsInfo> dimsInfo = {
        { seq_len, 2048, 2048, 44, },
        { seq_len, 256, 2048, 44, },
        { seq_len, 5632, 2048, 44, },
        { seq_len, 256, 64, 22, },
        { seq_len, 64, 256, 22, },
        { seq_len, 2048, 5632, 22, },
        { seq_len, 32000, 2048, 1, },
    };
    /*
    vector<DimsInfo> dimsInfo = {
        { 1, 4, 3, 1, },
    };
    */
    for (auto [m, n, k, c] : dimsInfo) {
        mul_mat_test_fp16(m, k, n);
    }
    return 0;
}

