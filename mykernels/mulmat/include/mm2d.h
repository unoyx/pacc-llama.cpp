#pragma once

#include <cassert>
#include <stdexcept>
#include <vector>

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

    Matrix2D(int row, int col, const std::vector<float> &data, enum Matrix2DLayout layout = ROW_MAJOR)
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

    std::vector<float> get_data() const {
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
    std::vector<float> data;
};
