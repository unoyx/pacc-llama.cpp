#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <stdexcept>
#include <vector>

#include "ggml.h"
#include "ggml-impl.h"
#include "sgemm.h"

using namespace std;

enum Matrix2DLayout
{
    ROW_MAJOR,
    COL_MAJOR,
};

struct Matrix2D
{

    Matrix2D(int row, int col, enum Matrix2DLayout layout = ROW_MAJOR)
    :layout(layout), m_row(row), m_col(col) {
        assert(m_row > 0);
        assert(m_col > 0);

        data.resize(m_row * m_col);
    }

    Matrix2D(int row, int col, vector<float> data, enum Matrix2DLayout layout = ROW_MAJOR)
    :layout(layout), m_row(row), m_col(col), data(data) {
        assert(m_row > 0);
        assert(m_col > 0);
        assert(data.size() == (m_row * m_col));
    }

    float& operator()(int row, int col) {
        if (row < 0 || row >= m_row || col < 0 || col >= m_col) {
            throw std::out_of_range("Index out of bounds");
        }
        if (layout == ROW_MAJOR) {
            return data[row * m_col + col];
        } else {
            return data[row + col * m_row];
        }
    }

    const float& operator()(int row, int col) const {
        if (row < 0 || row >= m_row || col < 0 || col >= m_col) {
            throw std::out_of_range("Index out of bounds");
        }
        if (layout == ROW_MAJOR) {
            return data[row * m_col + col];
        } else {
            return data[row + col * m_row];
        }
    }

    vector<float> get_data() const {
        return data;
    }

    void show() const {
        printf("row: %d, col: %d\n", m_row, m_col);
        for (int i = 0; i < m_row; ++i) {
            for (int j = 0; j < m_col; ++j) {
                float e = this->operator()(i, j);
                printf("%f, ", e);
            }
            printf("\n");
        }
    }

    int get_row() const {
        return m_row;
    }

    int get_col() const {
        return m_col;
    }

    enum Matrix2DLayout get_layout() const {
        return layout;
    }

    enum Matrix2DLayout layout = ROW_MAJOR;
    int m_row = 0;
    int m_col = 0;
    vector<float> data;
};

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

    mul_mat_bf16(m, n, k, mm_a_fp16.data(), k * 2, mm_b_fp16.data(), k * 2, mm_c_data, m * 4);

    Matrix2D c(m, n, mm_c, COL_MAJOR);

    c.show();
}

int main()
{
    simple_test_mm_row_col();
    return 0;
}
