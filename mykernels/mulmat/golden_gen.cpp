#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <vector>
#include <cmath>
#include <filesystem>

#include "ggml.h"
#include "ggml-impl.h"
#include "sgemm.h"

#include "mm2d.h"
#include "tools.h"

using namespace std;

static void simple_test_mm_row_col() {
    int m = 8;
    int n = 2;
    int k = 32;

    Matrix2D mm_a(m, k);
    for (int i = 0; i < mm_a.get_row(); ++i) {
        for (int j = 0; j < mm_a.get_col(); ++j) {
            mm_a(i, j) = i;
        }
    }
    mm_a.show();

    Matrix2D mm_b(k, n, COL_MAJOR);
    for (int i = 0; i < mm_b.get_row(); ++i) {
        for (int j = 0; j < mm_b.get_col(); ++j) {
            mm_b(i, j) = j;
        }
    }
    mm_b.show();

    vector<ggml_fp16_t> mm_a_fp16;
    for (auto e : mm_a.get_data()) {
        mm_a_fp16.push_back(ggml_compute_fp32_to_fp16(e));
    }

    vector<ggml_fp16_t> mm_b_fp16;
    for (auto e : mm_b.get_data()) {
        mm_b_fp16.push_back(ggml_compute_fp32_to_fp16(e));
    }

    vector<float> mm_c(m * n);
    float * mm_c_data = mm_c.data();

    bool is_calculate = llamafile_sgemm(m, n, k, mm_a_fp16.data(), k, mm_b_fp16.data(), k, mm_c_data, m, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32);

    printf("is_calculate: %d\n", is_calculate);

    Matrix2D c(m, n, mm_c, COL_MAJOR);

    for (int i = 0; i < m * n; ++i) {
        printf("%f, ", mm_c_data[i]);
    }
    printf("\n");

    c.show();
}

struct GoldenDims {
    int m;
    int n;
    int k;

public:
    GoldenDims(int m, int n, int k)
        :m(m), n(n), k(k) {}
};

static void gen_sequence_golden_data()
{
    int seq_len[] = {/* 1, */ 2, 10, 512, 1024};

    vector<GoldenDims> dims;
    for (auto len : seq_len) {
        dims.emplace_back(GoldenDims(128, len, 256));
        dims.emplace_back(GoldenDims(256, len, 128));
        dims.emplace_back(GoldenDims(3072, len, 3072));
    }
    string output_dir("seq_golden");
    if (filesystem::is_directory(output_dir)) {
        filesystem::remove_all(output_dir);
    }
    filesystem::create_directory(output_dir);

    for (GoldenDims dim : dims) {
        const int m = dim.m;
        const int n = dim.n;
        const int k = dim.k;

        Matrix2D mm_a(m, k);
        for (int i = 0; i < mm_a.get_row(); ++i) {
            for (int j = 0; j < mm_a.get_col(); ++j) {
                mm_a(i, j) = j;
            }
        }

        Matrix2D mm_b(k, n, COL_MAJOR);
        for (int i = 0; i < mm_b.get_row(); ++i) {
            for (int j = 0; j < mm_b.get_col(); ++j) {
                mm_b(i, j) = i;
            }
        }

        vector<ggml_fp16_t> mm_a_fp16;
        for (auto e : mm_a.get_data()) {
            mm_a_fp16.push_back(ggml_compute_fp32_to_fp16(e));
        }

        vector<ggml_fp16_t> mm_b_fp16;
        for (auto e : mm_b.get_data()) {
            mm_b_fp16.push_back(ggml_compute_fp32_to_fp16(e));
        }

        vector<float> mm_c(m * n);
        float * mm_c_data = mm_c.data();

        bool is_calculate = llamafile_sgemm(m, n, k, mm_a_fp16.data(), k, mm_b_fp16.data(), k, mm_c_data, m, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32);

        printf("is_calculate: %d, m: %d, n: %d, k: %d\n", is_calculate, m, n, k);
        if (!is_calculate) {
            throw std::runtime_error("mm calculate fail");
        }


        auto base_path = std::filesystem::path{output_dir} / (to_string(m) + "f16x" + to_string(n) + "f16x" + to_string(k) + "f32.bin");


        save_mm_golden(m, n, k,
                       mm_a_fp16.data(), 0,
                       mm_b_fp16.data(), 0,
                       mm_c_data, 0,
                       GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32, base_path);

        Matrix2D c(m, n, mm_c, COL_MAJOR);
    }
}

static void simple_test_mm_col_row() {
    int m = 8;
    int n = 2;
    int k = 32;

    Matrix2D mm_a(m, k);
    for (int i = 0; i < mm_a.get_row(); ++i) {
        for (int j = 0; j < mm_a.get_col(); ++j) {
            mm_a(i, j) = j;
        }
    }
    mm_a.show();

    Matrix2D mm_b(k, n, COL_MAJOR);
    for (int i = 0; i < mm_b.get_row(); ++i) {
        for (int j = 0; j < mm_b.get_col(); ++j) {
            mm_b(i, j) = i;
        }
    }
    mm_b.show();

    vector<ggml_fp16_t> mm_a_fp16;
    for (auto e : mm_a.get_data()) {
        mm_a_fp16.push_back(ggml_compute_fp32_to_fp16(e));
    }

    vector<ggml_fp16_t> mm_b_fp16;
    for (auto e : mm_b.get_data()) {
        mm_b_fp16.push_back(ggml_compute_fp32_to_fp16(e));
    }

    vector<float> mm_c(m * n);
    float * mm_c_data = mm_c.data();

    bool is_calculate = llamafile_sgemm(m, n, k, mm_a_fp16.data(), k, mm_b_fp16.data(), k, mm_c_data, m, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32);

    printf("is_calculate: %d\n", is_calculate);

    Matrix2D c(m, n, mm_c, COL_MAJOR);

    for (int i = 0; i < m * n; ++i) {
        printf("%f, ", mm_c_data[i]);
    }
    printf("\n");

    c.show();
}

int main()
{
    init_ggml_table_f32_f16();
    /*
    simple_test_mm_row_col();
    simple_test_mm_col_row();
    */
    gen_sequence_golden_data();

    return 0;
}

