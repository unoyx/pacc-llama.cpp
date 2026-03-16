#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>

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

float cosine_dist(vector<float> a, vector<float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same length.");
    }
    if (a.empty()) {
        throw std::invalid_argument("Vectors cannot be empty.");
    }

    double dot_product = 0.0;
    double norm1_sq = 0.0;
    double norm2_sq = 0.0;
    unsigned int length = a.size();

    // Calculate dot product and squared norms
    for (unsigned int i = 0u; i < length; ++i) {
        dot_product += a[i] * b[i];
        norm1_sq += a[i] * a[i];
        norm2_sq += b[i] * b[i];
    }

    // Calculate the norms (magnitudes)
    double norm1 = std::sqrt(norm1_sq);
    double norm2 = std::sqrt(norm2_sq);

    // Handle potential division by zero (zero vectors)
    if (norm1 == 0.0 || norm2 == 0.0) {
        // Depending on use case, one might return a specific value or throw an error.
        // Returning 1.0 (maximal distance in many scenarios) is a common choice for zero vectors.
        return 1.0;
    }

    // Calculate cosine similarity
    double cosine_similarity = dot_product / (norm1 * norm2);

    // Ensure the result is within the valid range [-1, 1] due to potential floating point inaccuracies
    cosine_similarity = std::clamp(cosine_similarity, -1.0, 1.0);

    // Cosine distance is 1 - cosine similarity, range [0, 2]
    return 1.0 - cosine_similarity;
}

#include <endian.h>
#include <filesystem>
#include <fstream>

void save_floats(vector<float> v, const string & f) {
    if (v.empty()) {
        throw std::runtime_error("save empty vector");
    }

    vector<uint32_t> data(v.size());
    for (int i = 0; i < v.size(); ++i) {
        data[i] = htole32(*((uint32_t *)(&v[i])));
    }

    auto path = std::filesystem::path{f};
    std::ofstream file{path, std::ios::binary};
    if (!file) {
        throw std::runtime_error("failed to open binary output file: " + path.string());
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

vector<float> load_floats(const string & f)
{
    auto path = std::filesystem::path{f};
    // Open the file in binary mode and go to the end
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + f);
        return {};
    }

    // Get the file size
    std::streampos fileSize = file.tellg();

    // Go back to the beginning of the file
    file.seekg(0, std::ios::beg);

    // Read the data into a vector
    std::vector<char> fileData(fileSize);
    if (!file.read(fileData.data(), fileSize)) { // Use .data() for the char* pointer
        throw std::runtime_error("Failed to read file: " + f);
    }

    if ((fileData.size() % sizeof(float)) != 0) {
        throw std::runtime_error("invalid file size");
    }

    std::vector<float> ret;
    ret.reserve(fileData.size() / sizeof(float));
    int i = 0;
    while (i < fileData.size()) {
        uint32_t e = le32toh(*(uint32_t *)(fileData.data() + i));
        float fe = *(float *)&e;
        ret.push_back(fe);
        i += sizeof(float);
    }

    return ret;
}

vector<float> bin2floats(vector<char> bdata) {
    if ((bdata.size() % sizeof(float)) != 0) {
        throw std::runtime_error("invalid file size");
    }

    std::vector<float> ret;
    ret.reserve(bdata.size() / sizeof(float));
    int i = 0;
    while (i < bdata.size()) {
        uint32_t e = le32toh(*(uint32_t *)(bdata.data() + i));
        float fe = *(float *)&e;
        ret.push_back(fe);
        i += sizeof(float);
    }

    return ret;
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

    bool is_calculate = llamafile_sgemm(m, n, k, mm_a_fp16.data(), k, mm_b_fp16.data(), k, mm_c_data, m, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32);

    printf("is_calculate: %d\n", is_calculate);

    Matrix2D c(m, n, mm_c, COL_MAJOR);

    for (int i = 0; i < m * n; ++i) {
        printf("%f, ", mm_c_data[i]);
    }
    printf("\n");

    c.show();
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

static vector<float> to_floats(const void * ptr, int n, int type) {
    vector<float> ret;
    // float
    if (type == 0) {
        const auto * pp = (const float *)ptr;
        for (int i = 0; i < n; ++i) {
            ret.push_back(pp[i]);
        }
    // bf16
    } else if (type == 1) {
        const auto * pp = (const ggml_bf16_t *)ptr;
        for (int i = 0; i < n; ++i) {
            ret.push_back(ggml_compute_bf16_to_fp32(pp[i]));
        }
    } else if (type == 2) {
        const auto * pp = (const ggml_fp16_t *)ptr;
        for (int i = 0; i < n; ++i) {
            ret.push_back(ggml_compute_fp16_to_fp32(pp[i]));
        }
    }
    return ret;
}

static vector<uint32_t> floats2bin(vector<float> f) {
    vector<uint32_t> data(f.size());
    for (int i = 0; i < f.size(); ++i) {
        data[i] = htole32(*((uint32_t *)(&f[i])));
    }
    return data;
}

static void save_mm_golden(int64_t m, int64_t n, int64_t k,
                     const void *A, int64_t lda,
                     const void *B, int64_t ldb,
                     const void *C, int64_t ldc,
                     int32_t Atype, int32_t Btype, int32_t Ctype, string file_name) {
    vector<uint32_t> tA = floats2bin(to_floats(A, m * k, Atype));
    vector<uint32_t> tB = floats2bin(to_floats(B, n * k, Btype));
    vector<uint32_t> tC = floats2bin(to_floats(C, m * n, Ctype));

    auto path = std::filesystem::path{file_name};
    std::ofstream file{path, std::ios::binary};
    uint64_t bm = htole64(m);
    uint64_t bn = htole64(n);
    uint64_t bk = htole64(k);
    file.write(reinterpret_cast<const char*>(&bm), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&bn), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&bk), sizeof(uint64_t));
    uint32_t bAtype = htole32(Atype);
    uint32_t bBtype = htole32(Btype);
    uint32_t bCtype = htole32(Ctype);
    file.write(reinterpret_cast<const char*>(&bAtype), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&bBtype), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&bCtype), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(tA.data()), sizeof(uint32_t) * tA.size());
    file.write(reinterpret_cast<const char*>(tB.data()), sizeof(uint32_t) * tB.size());
    file.write(reinterpret_cast<const char*>(tC.data()), sizeof(uint32_t) * tC.size());
}

static void load_mm_golden(string file_name,
                           int64_t *pm, int64_t *pn, int64_t *pk,
                           int32_t *pAtype, int32_t *pBtype, int32_t *pCtype,
                           vector<float> *pA, vector<float> *pB, vector<float> *pC) {
    assert(pm != nullptr);
    assert(pn != nullptr);
    assert(pk != nullptr);
    assert(pAtype != nullptr);
    assert(pBtype != nullptr);
    assert(pCtype != nullptr);
    assert(pA != nullptr);
    assert(pB != nullptr);
    assert(pC != nullptr);

    auto path = std::filesystem::path{file_name};
    std::ifstream file{path, std::ios::binary};

    uint64_t t64;

    file.read((char *)&t64, sizeof(uint64_t));
    int64_t m = le64toh(t64);
    *pm = m;

    file.read((char *)&t64, sizeof(uint64_t));
    int64_t n = le64toh(t64);
    *pn = n;

    file.read((char *)&t64, sizeof(uint64_t));
    int64_t k = le64toh(t64);
    *pk = k;

    uint32_t t32;

    file.read((char *)&t32, sizeof(uint32_t));
    int32_t Atype = le32toh(t32);
    *pAtype = Atype;

    file.read((char *)&t32, sizeof(uint32_t));
    int32_t Btype = le32toh(t32);
    *pBtype = Btype;

    file.read((char *)&t32, sizeof(uint32_t));
    int32_t Ctype = le32toh(t32);
    *pCtype = Ctype;

    vector<char> bA(m * k * sizeof(float));
    file.read(bA.data(), bA.size());
    vector<float> vA = bin2floats(bA);
    *pA = vA;

    vector<char> bB(n * k * sizeof(float));
    file.read(bB.data(), bB.size());
    vector<float> vB = bin2floats(bB);
    *pB = vB;

    vector<char> bC(m * n * sizeof(float));
    file.read(bC.data(), bC.size());
    vector<float> vC = bin2floats(bC);
    *pC = vC;
}

float ggml_table_f32_f16[1 << 16];

void init_ggml_table_f32_f16()
{
    static bool first_run = true;
    if (first_run) {
        for (int i = 0; i < (1 << 16); ++i) {
            union {
                uint16_t u16;
                ggml_fp16_t fp16;
            } u = {(uint16_t)i};
            float f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
            ggml_table_f32_f16[i] = f;
        }
    }
}


int main()
{
    init_ggml_table_f32_f16();
    /*
    simple_test_mm_row_col();
    simple_test_mm_col_row();
    */

    int m = 2;
    int n = 3;
    int k = 4;

    vector<float> va(m * k);
    for (int i = 0; i < va.size(); ++i) {
        va.at(i) = i;
    }
    vector<float> vb(n * k);
    for (int i = 0; i < vb.size(); ++i) {
        vb.at(i) = i;
    }
    vector<float> vc(m * n);
    for (int i = 0; i < vc.size(); ++i) {
        vc.at(i) = i;
    }

    save_mm_golden(m, n, k, va.data(), 0, vb.data(), 0, vc.data(), 0, 0, 0, 0, "golden.bin");

    int64_t gm;
    int64_t gn;
    int64_t gk;

    int32_t ta;
    int32_t tb;
    int32_t tc;

    vector<float> gA;
    vector<float> gB;
    vector<float> gC;

    load_mm_golden("golden.bin",
                           &gm, &gn, &gk,
                           &ta, &tb, &tc,
                           &gA, &gB, &gC);

    printf("A: ");
    for (auto e : gA) {
        printf("%f, ", e);
    }
    printf("%d\n", gA.size());

    printf("B: ");
    for (auto e : gB) {
        printf("%f, ", e);
    }
    printf("%d\n", gB.size());

    printf("C: ");
    for (auto e : gC) {
        printf("%f, ", e);
    }
    printf("%d\n", gC.size());


    return 0;
}

