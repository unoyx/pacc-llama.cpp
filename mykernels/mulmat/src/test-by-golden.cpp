#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <filesystem>
#include <numeric>

#include "ggml.h"
#include "ggml-impl.h"

#include "sgemm.h"

#include "mm2d.h"
#include "tools.h"

using namespace std;

static void simple_golden_test(std::filesystem::path dir) {

    if (!std::filesystem::is_directory(dir)) {
        throw std::runtime_error("not a directory");
    }

    auto iter = std::filesystem::directory_iterator{dir};

    for (auto const& dir_entry : iter)  {
        if (!dir_entry.is_regular_file()) {
            continue;
        }

        int64_t m = 0;
        int64_t n = 0;
        int64_t k = 0;

        int32_t Atype = 0;
        int32_t Btype = 0;
        int32_t Ctype = 0;

        std::vector<float> mA;
        std::vector<float> mB;
        std::vector<float> mC;
        load_mm_golden(dir_entry,
                       &m, &n, &k,
                       &Atype, &Btype, &Ctype,
                       &mA, &mB, &mC);

        vector<ggml_fp16_t> mm_a_fp16;
        for (auto e : mA) {
            mm_a_fp16.push_back(ggml_compute_fp32_to_fp16(e));
        }

        vector<ggml_fp16_t> mm_b_fp16;
        for (auto e : mB) {
            mm_b_fp16.push_back(ggml_compute_fp32_to_fp16(e));
        }

        std::vector<float> calC(m * n);
        // mul_mat_bf16(m, n, k, mm_a_fp16.data(), k * 2, mm_b_fp16.data(), k * 2, calC.data(), m * 4);
        bool ret = llamafile_sgemm(m, n, k,
                     mm_a_fp16.data(), k, mm_b_fp16.data(), k, calC.data(), m,
                     GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32);

        float dis = cosine_dist(mC, calC);
        printf("m:%d, n:%d, k:%d, cosdist: %f, calculated: %d\n", m, n, k, dis, ret);
        if (dis > 0) {
            /*
            double sum_of_golden = std::accumulate(mC.begin(), mC.end(), 0);
            double sum_of_result = std::accumulate(calC.begin(), calC.end(), 0);
            printf("sum_of_golden: %f\n", sum_of_golden);
            printf("sum_of_result: %f\n", sum_of_result);

            Matrix2D mm_a(m, k, mA);
            mm_a.show();

            Matrix2D mm_b(m, k, mB, COL_MAJOR);
            mm_b.show();

            Matrix2D c_golden(m, n, mC, COL_MAJOR);
            printf("_golden:\n");
            c_golden.show();
            Matrix2D c_result(m, n, calC, COL_MAJOR);
            printf("_result:\n");
            c_result.show();
            */
            return;
        }
    }
}


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

    // mul_mat_bf16(m, n, k, mm_a_fp16.data(), k * 2, mm_b_fp16.data(), k * 2, mm_c_data, m * 4);

    Matrix2D c(m, n, mm_c, COL_MAJOR);

    c.show();
}

int main()
{
    init_ggml_table_f32_f16();
    simple_golden_test("./seq_golden");
    return 0;
}
