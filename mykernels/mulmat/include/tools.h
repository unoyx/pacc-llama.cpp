#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>

float cosine_dist(std::vector<float> a, std::vector<float> b);

void save_floats(std::vector<float> v, const std::string & f);

std::vector<float> load_floats(const std::string & f);

std::vector<float> bin2floats(std::vector<char> bdata);

std::vector<float> to_floats(const void * ptr, int n, int type);

std::vector<uint32_t> floats2bin(std::vector<float> f);

void save_mm_golden(int64_t m, int64_t n, int64_t k,
                     const void *A, int64_t lda,
                     const void *B, int64_t ldb,
                     const void *C, int64_t ldc,
                     int32_t Atype, int32_t Btype, int32_t Ctype, std::filesystem::path file_name);

void load_mm_golden(std::filesystem::path file_path,
                           int64_t *pm, int64_t *pn, int64_t *pk,
                           int32_t *pAtype, int32_t *pBtype, int32_t *pCtype,
                           std::vector<float> *pA, std::vector<float> *pB, std::vector<float> *pC);

extern float ggml_table_f32_f16[1 << 16];

void init_ggml_table_f32_f16();
