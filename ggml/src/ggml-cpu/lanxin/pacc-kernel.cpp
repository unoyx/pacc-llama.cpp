
#include "pacc.h"

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


namespace ggml::cpu::riscv64_pacc_kernel { 
//激活矩阵[m,k],行优先顺序，
//权重矩阵[k,n],列主序顺序，reapck为
//权重矩阵[k/tile_k,n/tile_n,tile_k,tile_n]
//tile内部行优先
//tile外部之间列优先排列
//结果矩阵[m,n],行优先顺序，
//要求输入A，B，C内存布局连续
//断言vlen长度为1024
//计算gemv
template <int layout_block_n>
static inline void micro_kernel_fp16fp16fp32_tile_k1_tile_n_gemv(int m_v, int n_v, int k_v,  const _Float16* A, const _Float16* B, float* C) {
    constexpr int block_n = layout_block_n;
    assert(m_v <= 4);
    if(m_v == 1){
        for (int j = 0; j < n_v;) {
            size_t vl = __riscv_vsetvl_e16m4(n_v - j);
            assert(vl<=block_n);
            vfloat32m8_t vc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            const _Float16 *lhs_tile = &A[0];
            const _Float16 *rhs_tile = &B[j*k_v];
            int k = 0;
            // 主循环，每次处理4个k维度激活元素
            for (; k < k_v-3; k+=4) {
                _Float16 a0 = lhs_tile[k+0];
                _Float16 a1 = lhs_tile[k+1];
                _Float16 a2 = lhs_tile[k+2];
                _Float16 a3 = lhs_tile[k+3];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(rhs_tile+0*vl, vl); // 256x2x4 = 2048 Bytes fp16 Load               
		vfloat16m4_t vb1 = __riscv_vle16_v_f16m4(rhs_tile+1*vl, vl);
                vfloat16m4_t vb2 = __riscv_vle16_v_f16m4(rhs_tile+2*vl, vl);
                vfloat16m4_t vb3 = __riscv_vle16_v_f16m4(rhs_tile+3*vl, vl);
                vc = __riscv_vfwmacc_vf_f32m8(vc, a0, vb0, vl);
		vc = __riscv_vfwmacc_vf_f32m8(vc, a1, vb1, vl);
                vc = __riscv_vfwmacc_vf_f32m8(vc, a2, vb2, vl);
                vc = __riscv_vfwmacc_vf_f32m8(vc, a3, vb3, vl);
                rhs_tile += 4*vl;
            }
            for (; k < k_v; ++k) {
                _Float16 a = lhs_tile[k];
                vfloat16m4_t vb = __riscv_vle16_v_f16m4(rhs_tile, vl);
                rhs_tile += vl;
                vc = __riscv_vfwmacc_vf_f32m8(vc, a, vb, vl);
            }
            __riscv_vse32_v_f32m8(&C[j], vc, vl);
            j += vl;
        }
    } else if (m_v == 2){
        for (int j = 0; j < n_v;) {
            size_t vl = __riscv_vsetvl_e16m4(n_v - j);
            assert(vl<=block_n);
            vfloat32m8_t vc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc1 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            const _Float16 *lhs_tile0 = &A[0];
            const _Float16 *lhs_tile1 = &A[k_v];
            const _Float16 *rhs_tile = &B[j*k_v];
            int k = 0;
            // 主循环，每次处理4个k维度激活元素
            for (; k < k_v-1; k+=2) {
                _Float16 a00 = lhs_tile0[k+0];
                _Float16 a01 = lhs_tile0[k+1];
                _Float16 a10 = lhs_tile1[k+0];
                _Float16 a11 = lhs_tile1[k+1];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(rhs_tile+0*vl, vl);
                vfloat16m4_t vb1 = __riscv_vle16_v_f16m4(rhs_tile+1*vl, vl);
                vc0 = __riscv_vfwmacc_vf_f32m8(vc0, a00, vb0, vl);
                vc1 = __riscv_vfwmacc_vf_f32m8(vc1, a10, vb0, vl);
                vc0 = __riscv_vfwmacc_vf_f32m8(vc0, a01, vb1, vl);
                vc1 = __riscv_vfwmacc_vf_f32m8(vc1, a11, vb1, vl);
                rhs_tile += 2*vl;
            }
            if(k!=k_v){
                _Float16 a00 = lhs_tile0[k];
                _Float16 a10 = lhs_tile1[k];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(rhs_tile, vl);
                vc0 = __riscv_vfwmacc_vf_f32m8(vc0, a00, vb0, vl);
                vc1 = __riscv_vfwmacc_vf_f32m8(vc1, a10, vb0, vl);
            }
            __riscv_vse32_v_f32m8(&C[j], vc0, vl);
            __riscv_vse32_v_f32m8(&C[k_v+j], vc1, vl);
            j += vl;
        }
    } else if (m_v == 3){
        for (int j = 0; j < n_v;) {
            size_t vl = __riscv_vsetvl_e16m4(n_v - j);
            assert(vl<=block_n);
            vfloat32m8_t vc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc1 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc2 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            const _Float16 *lhs_tile0 = &A[0];
            const _Float16 *lhs_tile1 = &A[k_v];
            const _Float16 *lhs_tile2 = &A[2*k_v];
            const _Float16 *rhs_tile = &B[j*k_v];
            int k = 0;
            // 主循环，每次处理4个k维度激活元素
            for (; k < k_v-1; k+=2) {
                _Float16 a00 = lhs_tile0[k];
                _Float16 a01 = lhs_tile0[k+1];
                _Float16 a10 = lhs_tile1[k];
                _Float16 a11 = lhs_tile1[k+1];
                _Float16 a20 = lhs_tile2[k];
                _Float16 a21 = lhs_tile2[k+1];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(rhs_tile, vl);
                vfloat16m4_t vb1 = __riscv_vle16_v_f16m4(rhs_tile+vl, vl);
                vc0 = __riscv_vfwmacc_vf_f32m8(vc0, a00, vb0, vl);
                vc1 = __riscv_vfwmacc_vf_f32m8(vc1, a10, vb0, vl);
                vc2 = __riscv_vfwmacc_vf_f32m8(vc2, a20, vb0, vl);
                vc0 = __riscv_vfwmacc_vf_f32m8(vc0, a01, vb1, vl);
                vc1 = __riscv_vfwmacc_vf_f32m8(vc1, a11, vb1, vl);
                vc2 = __riscv_vfwmacc_vf_f32m8(vc2, a21, vb1, vl);
            }
            if(k!=k_v){
                _Float16 a00 = lhs_tile0[k];
                _Float16 a10 = lhs_tile1[k];
                _Float16 a20 = lhs_tile2[k];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(rhs_tile, vl);
                vc0 = __riscv_vfwmacc_vf_f32m8(vc0, a00, vb0, vl);
                vc1 = __riscv_vfwmacc_vf_f32m8(vc1, a10, vb0, vl);
                vc2 = __riscv_vfwmacc_vf_f32m8(vc2, a20, vb0, vl);
                rhs_tile += vl;
            }
            __riscv_vse32_v_f32m8(&C[j], vc0, vl);
            __riscv_vse32_v_f32m8(&C[k_v+j], vc1, vl);
            __riscv_vse32_v_f32m8(&C[2*k_v+j], vc2, vl);
            j += vl;
        }
    } else if (m_v == 4){
        for (int j = 0; j < n_v;) {
            int stride = min(n_v - j, 256);
            size_t vl = __riscv_vsetvl_e16m2(n_v - j);
            assert(vl<=block_n/2);
            vfloat32m8_t vc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m8_t vc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m8_t vc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m8_t vc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            const _Float16 *lhs_tile0 = &A[0];
            const _Float16 *lhs_tile1 = &A[k_v];
            const _Float16 *lhs_tile2 = &A[2*k_v];
            const _Float16 *lhs_tile3 = &A[3*k_v];
            const _Float16 *rhs_tile  = &B[j*k_v]; 
            if((j/128)%2 == 1){
                *rhs_tile  = &B[j*k_v + 128]; 
                if((n_v - j) < 128){
                    stride += 128;
                }
            }
            int k = 0;
            // 主循环，每次处理4个k维度激活元素
            for (; k < k_v-1; k+=2) {
                _Float16 a00 = lhs_tile0[k];
                _Float16 a01 = lhs_tile0[k+1];
                _Float16 a10 = lhs_tile1[k];
                _Float16 a11 = lhs_tile1[k+1];
                _Float16 a20 = lhs_tile2[k];
                _Float16 a21 = lhs_tile2[k+1];
                _Float16 a30 = lhs_tile3[k];
                _Float16 a31 = lhs_tile3[k+1];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m2(rhs_tile, vl);
                vfloat16m4_t vb1 = __riscv_vle16_v_f16m2(rhs_tile + stride, vl);
                vc0 = __riscv_vfwmacc_vf_f32m4(vc0, a00, vb0, vl);
                vc1 = __riscv_vfwmacc_vf_f32m4(vc1, a10, vb0, vl);
                vc2 = __riscv_vfwmacc_vf_f32m4(vc2, a20, vb0, vl);
                vc3 = __riscv_vfwmacc_vf_f32m4(vc3, a30, vb0, vl);
                vc0 = __riscv_vfwmacc_vf_f32m4(vc0, a01, vb1, vl);
                vc1 = __riscv_vfwmacc_vf_f32m4(vc1, a11, vb1, vl);
                vc2 = __riscv_vfwmacc_vf_f32m4(vc2, a21, vb1, vl);
                vc3 = __riscv_vfwmacc_vf_f32m4(vc3, a31, vb1, vl);
                rhs_tile += 2*stride;
            }
            if(k!=k_v){
                _Float16 a00 = lhs_tile0[k];
                _Float16 a10 = lhs_tile1[k];
                _Float16 a20 = lhs_tile2[k];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(rhs_tile, vl);
                vc0 = __riscv_vfwmacc_vf_f32m4(vc0, a00, vb0, vl);
                vc1 = __riscv_vfwmacc_vf_f32m4(vc1, a10, vb0, vl);
                vc2 = __riscv_vfwmacc_vf_f32m4(vc2, a20, vb0, vl);
                rhs_tile += vl;
            }
            __riscv_vse32_v_f32m4(&C[j], vc0, vl);
            __riscv_vse32_v_f32m4(&C[k_v+j], vc1, vl);
            __riscv_vse32_v_f32m4(&C[2*k_v+j], vc2, vl);
            __riscv_vse32_v_f32m4(&C[2*k_v+j], vc2, vl);
            j += vl;
        }
    } else {
        assert(m_v == 0);
    }
}


template <int layout_block_n>
static inline void micro_kernel_bf16bf16fp32_tile_k1_tile_n_gemv(int m_v, int n_v, int k_v,  const __bf16* A, const __bf16* B, float* C) {
    constexpr int block_n = layout_block_n;
    assert(m_v <= 4);
    if(m_v == 1){
        for (int j = 0; j < n_v;) {
            size_t vl = __riscv_vsetvl_e16m4(n_v - j);
            assert(vl<=block_n);
            vfloat32m8_t vc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            const __bf16 *lhs_tile = &A[0];
            const __bf16 *rhs_tile = &B[j*k_v];
            int k = 0;
            // 主循环，每次处理4个k维度激活元素
            for (; k < k_v-3; k+=4) {
                __bf16 a0 = lhs_tile[k+0];
                __bf16 a1 = lhs_tile[k+1];
                __bf16 a2 = lhs_tile[k+2];
                __bf16 a3 = lhs_tile[k+3];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(rhs_tile+0*vl, vl); // 256x2x4 = 2048 Bytes fp16 Load B
                vfloat16m4_t vb1 = __riscv_vle16_v_f16m4(rhs_tile+1*vl, vl);
                vfloat16m4_t vb2 = __riscv_vle16_v_f16m4(rhs_tile+2*vl, vl);
                vfloat16m4_t vb3 = __riscv_vle16_v_f16m4(rhs_tile+3*vl, vl);
                vc = __riscv_vfwmaccbf16_vv_f32m8(vc, a0, vb0, vl);
                vc = __riscv_vfwmaccbf16_vv_f32m8(vc, a1, vb1, vl);
                vc = __riscv_vfwmaccbf16_vv_f32m8(vc, a2, vb2, vl);
                vc = __riscv_vfwmaccbf16_vv_f32m8(vc, a3, vb3, vl);
                rhs_tile += 4*vl;
            }
            for (; k < k_v; ++k) {
                __bf16 a = lhs_tile[k];
                vfloat16m4_t vb = __riscv_vle16_v_f16m4(rhs_tile, vl);
                rhs_tile += vl;
                vc = __riscv_vfwmaccbf16_vv_f32m8(vc, a, vb, vl);
            }
            __riscv_vse32_v_f32m8(&C[j], vc, vl);
            j += vl;
        }
    } else if (m_v == 2){
        for (int j = 0; j < n_v;) {
            size_t vl = __riscv_vsetvl_e16m4(n_v - j);
            assert(vl<=block_n);
            vfloat32m8_t vc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc1 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            const __bf16 *lhs_tile0 = &A[0];
            const __bf16 *lhs_tile1 = &A[k_v];
            const __bf16 *rhs_tile = &B[j*k_v];
            int k = 0;
            // 主循环，每次处理4个k维度激活元素
            for (; k < k_v-1; k+=2) {
                __bf16 a00 = lhs_tile0[k+0];
                __bf16 a01 = lhs_tile0[k+1];
                __bf16 a10 = lhs_tile1[k+0];
                __bf16 a11 = lhs_tile1[k+1];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(rhs_tile+0*vl, vl);
                vfloat16m4_t vb1 = __riscv_vle16_v_f16m4(rhs_tile+1*vl, vl);
                vc0 = __riscv_vfwmaccbf16_vv_f32m8(vc0, a00, vb0, vl);
                vc1 = __riscv_vfwmaccbf16_vv_f32m8(vc1, a10, vb0, vl);
                vc0 = __riscv_vfwmaccbf16_vv_f32m8(vc0, a01, vb1, vl);
                vc1 = __riscv_vfwmaccbf16_vv_f32m8(vc1, a11, vb1, vl);
                rhs_tile += 2*vl;
            }
            if(k!=k_v){
                __bf16 a00 = lhs_tile0[k];
                __bf16 a10 = lhs_tile1[k];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(rhs_tile, vl);
                vc0 = __riscv_vfwmaccbf16_vv_f32m8(vc0, a00, vb0, vl);
                vc1 = __riscv_vfwmaccbf16_vv_f32m8(vc1, a10, vb0, vl);
            }
            __riscv_vse32_v_f32m8(&C[j], vc0, vl);
            __riscv_vse32_v_f32m8(&C[k_v+j], vc1, vl);
            j += vl;
        }
    } else if (m_v == 3){
        for (int j = 0; j < n_v;) {
            size_t vl = __riscv_vsetvl_e16m4(n_v - j);
            assert(vl<=block_n);
            vfloat32m8_t vc0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc1 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vfloat32m8_t vc2 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            const __bf16 *lhs_tile0 = &A[0];
            const __bf16 *lhs_tile1 = &A[k_v];
            const __bf16 *lhs_tile2 = &A[2*k_v];
            const __bf16 *rhs_tile = &B[j*k_v];
            int k = 0;
            // 主循环，每次处理4个k维度激活元素
            for (; k < k_v-1; k+=2) {
                __bf16 a00 = lhs_tile0[k];
                __bf16 a01 = lhs_tile0[k+1];
                __bf16 a10 = lhs_tile1[k];
                __bf16 a11 = lhs_tile1[k+1];
                __bf16 a20 = lhs_tile2[k];
                __bf16 a21 = lhs_tile2[k+1];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(rhs_tile, vl);
                vfloat16m4_t vb1 = __riscv_vle16_v_f16m4(rhs_tile+vl, vl);
                vc0 = __riscv_vfwmaccbf16_vv_f32m8(vc0, a00, vb0, vl);
                vc1 = __riscv_vfwmaccbf16_vv_f32m8(vc1, a10, vb0, vl);
                vc2 = __riscv_vfwmaccbf16_vv_f32m8(vc2, a20, vb0, vl);
                vc0 = __riscv_vfwmaccbf16_vv_f32m8(vc0, a01, vb1, vl);
                vc1 = __riscv_vfwmaccbf16_vv_f32m8(vc1, a11, vb1, vl);
                vc2 = __riscv_vfwmaccbf16_vv_f32m8(vc2, a21, vb1, vl);
            }
            if(k!=k_v){
                __bf16 a00 = lhs_tile0[k];
                __bf16 a10 = lhs_tile1[k];
                __bf16 a20 = lhs_tile2[k];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(rhs_tile, vl);
                vc0 = __riscv_vfwmaccbf16_vv_f32m8(vc0, a00, vb0, vl);
                vc1 = __riscv_vfwmaccbf16_vv_f32m8(vc1, a10, vb0, vl);
                vc2 = __riscv_vfwmaccbf16_vv_f32m8(vc2, a20, vb0, vl);
                rhs_tile += vl;
            }
            __riscv_vse32_v_f32m8(&C[j], vc0, vl);
            __riscv_vse32_v_f32m8(&C[k_v+j], vc1, vl);
            __riscv_vse32_v_f32m8(&C[2*k_v+j], vc2, vl);
            j += vl;
        }
    } else if (m_v == 4){
        for (int j = 0; j < n_v;) {
            int stride = min(n_v - j, 256);
            size_t vl = __riscv_vsetvl_e16m2(n_v - j);
            assert(vl<=block_n/2);
            vfloat32m8_t vc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m8_t vc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m8_t vc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m8_t vc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            const __bf16 *lhs_tile0 = &A[0];
            const __bf16 *lhs_tile1 = &A[k_v];
            const __bf16 *lhs_tile2 = &A[2*k_v];
            const __bf16 *lhs_tile3 = &A[3*k_v];
            const __bf16 *rhs_tile  = &B[j*k_v]; 
            if((j/128)%2 == 1){
                *rhs_tile  = &B[j*k_v + 128]; 
                if((n_v - j) < 128){
                    stride += 128;
                }
            }
            int k = 0;
            // 主循环，每次处理4个k维度激活元素
            for (; k < k_v-1; k+=2) {
                __bf16 a00 = lhs_tile0[k];
                __bf16 a01 = lhs_tile0[k+1];
                __bf16 a10 = lhs_tile1[k];
                __bf16 a11 = lhs_tile1[k+1];
                __bf16 a20 = lhs_tile2[k];
                __bf16 a21 = lhs_tile2[k+1];
                __bf16 a30 = lhs_tile3[k];
                __bf16 a31 = lhs_tile3[k+1];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m2(rhs_tile, vl);
                vfloat16m4_t vb1 = __riscv_vle16_v_f16m2(rhs_tile + stride, vl);
                vc0 = __riscv_vfwmaccbf16_vv_f32m4(vc0, a00, vb0, vl);
                vc1 = __riscv_vfwmaccbf16_vv_f32m4(vc1, a10, vb0, vl);
                vc2 = __riscv_vfwmaccbf16_vv_f32m4(vc2, a20, vb0, vl);
                vc3 = __riscv_vfwmaccbf16_vv_f32m4(vc3, a30, vb0, vl);
                vc0 = __riscv_vfwmaccbf16_vv_f32m4(vc0, a01, vb1, vl);
                vc1 = __riscv_vfwmaccbf16_vv_f32m4(vc1, a11, vb1, vl);
                vc2 = __riscv_vfwmaccbf16_vv_f32m4(vc2, a21, vb1, vl);
                vc3 = __riscv_vfwmaccbf16_vv_f32m4(vc3, a31, vb1, vl);
                rhs_tile += 2*stride;
            }
            if(k!=k_v){
                __bf16 a00 = lhs_tile0[k];
                __bf16 a10 = lhs_tile1[k];
                __bf16 a20 = lhs_tile2[k];
                vfloat16m4_t vb0 = __riscv_vle16_v_f16m4(rhs_tile, vl);
                vc0 = __riscv_vfwmaccbf16_vv_f32m4(vc0, a00, vb0, vl);
                vc1 = __riscv_vfwmaccbf16_vv_f32m4(vc1, a10, vb0, vl);
                vc2 = __riscv_vfwmacc_vf_f32m4(vc2, a20, vb0, vl);
                rhs_tile += vl;
            }
            __riscv_vse32_v_f32m4(&C[j], vc0, vl);
            __riscv_vse32_v_f32m4(&C[k_v+j], vc1, vl);
            __riscv_vse32_v_f32m4(&C[2*k_v+j], vc2, vl);
            __riscv_vse32_v_f32m4(&C[2*k_v+j], vc2, vl);
            j += vl;
        }
    } else {
        assert(m_v == 0);
    }
}


//A是激活
//B是权重
// [DK, 256]
// [DK]
// [DK / 32]
// [DK / 32, 256]
// [256]  输出累加器
template <int layout_block_n>
static inline void micro_kernel_q8_0_q8_0fp32_tile_k32_tile_n_gemv(int m_v, int n_v, int k_v,  const block_q8_0 A, const int8_t* B, float* C) {
    constexpr int block_n = layout_block_n;
    const size_t BLOCK_SIZE = 32;
    const size_t num_blocks = DK / BLOCK_SIZE;
    const int k_stride = num_blocks * sizeof(block_q8_0);

    assert(m_v <= 4 && m_v >= 1);
    for(int i = 0;i < m_v;i++){
        for (int j = 0; j < n_v;) {
            int re = MIN(n_v - j, block_n)
            size_t vl = __riscv_vsetvl_e16m4(re);
            vfloat32m8_t vout = __riscv_vfmv_v_f_f32m8(0.0f, vl);//8
            assert(vl<=block_n);
            const block_q8_0 *lhs_tile = A                  ;
            const int8_t     *rhs_tile = B + (j * k_stride) ;

            for (size_t block = 0; block < num_blocks; ++block) {
                const _Float16 *w_base_s = (_Float16 *)(rhs_tile + block * vl * sizeof(block_q8_0));
                const int8_t   *w_base_q = (int8_t   *)(w_base_s + vl);
                const block_q8_0 *a_base = lhs_tile + block;
                vint32m8_t vacc = __riscv_vmv_v_i_i32m8(0, vl);//8
                for (size_t kk = 0; kk < BLOCK_SIZE; kk+=4) {
                    int8_t aq_val0 = a_base->qs[kk+0];
                    int8_t aq_val1 = a_base->qs[kk+1];
                    int8_t aq_val2 = a_base->qs[kk+2];
                    int8_t aq_val3 = a_base->qs[kk+3];
                    vint8m2_t vwq0 = __riscv_vle8_v_i8m2(w_base_q + (kk+0) * vl, vl);//8
                    vint8m2_t vwq1 = __riscv_vle8_v_i8m2(w_base_q + (kk+1) * vl, vl);
                    vint8m2_t vwq2 = __riscv_vle8_v_i8m2(w_base_q + (kk+2) * vl, vl);
                    vint8m2_t vwq3 = __riscv_vle8_v_i8m2(w_base_q + (kk+3) * vl, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, aq_val0, vwq0, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, aq_val1, vwq1, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, aq_val2, vwq2, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, aq_val3, vwq3, vl);
                }
                _Float16 aqs_val = a_base->d;
                vfloat16m4_t vwqs = __riscv_vle16_v_f16m4(w_base_s, vl);
                vfloat32m8_t vscale = __riscv_vfwmul_vf_f32m8(vwqs, aqs_val, vl);
                vfloat32m8_t vf_acc = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
                vout = __riscv_vfmacc_vv_f32m8(vout, vf_acc, vscale, vl);
            }
            __riscv_vse32_v_f32m8(C_fp32, vout, vl);
            j += vl;
        }   
    }
}
//激活矩阵[m,k],行优先顺序，
//权重矩阵[k,n],行优先顺序，reapck为
//权重矩阵[k/tile_k,n/tile_n,tile_k,tile_n]
//tile内部行优先
//tile外部之间列优先排列
//结果矩阵[m,n],行优先顺序，
//要求输入A，B，C内存布局连续
//断言vlen长度为1024
//计算gemv

template <int layout_block_n>
static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    constexpr int block_n = layout_block_n;
    constexpr int block_m = 4  ;
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];


    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);
    const bool dst_cont = ggml_is_contiguous(dst);

    assert(ir0_start % block_n == 0);

    assert(src1_cont == true);
    assert(dst_cont == true);
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    bool is_fp16_type = vec_dot_type == GGML_TYPE_F16;
    bool is_bf16_type = vec_dot_type == GGML_TYPE_BF16;
    bool is_q8_0_type = vec_dot_type == GGML_TYPE_Q8_0;
    assert(is_fp16_type || is_bf16_type || is_q8_0_type);

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

   const size_t row_size = ggml_row_size(vec_dot_type, ne10); 
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += block_m) {
        const char * src0_row = (const char*)src0->data + (ir0_start * src0->nb[1]);                             //weight
        const char * src1_col = (const char*)wdata + (iir1 * row_size);                                     //activate
        float * dst_col = (float*)((char*)dst->data + (ir0_start * sizeof(float)));                              //result
        size_t tile_k_v = ne00;
        size_t tile_n_v = ir0_end - ir0_start;
        size_t tile_m_v = MIN(ir1_end-iir1, block_m);
        if(is_bf16_type){
            micro_kernel_bf16bf16fp32_tile_k1_tile_n_gemv<block_n>(tile_m_v, tile_n_v, tile_k_v, (__bf16*)src1_col, (__bf16*)src0_row, dst_col);
        } else if(is_fp16_type) {
            micro_kernel_fp16fp16fp32_tile_k1_tile_n_gemv<block_n>(tile_m_v, tile_n_v, tile_k_v, (_Float16*)src1_col, (_Float16*)src0_row, dst_col);
        } else {
            micro_kernel_q8_0_q8_0fp32_tile_k32_tile_n_gemv<block_n>(tile_m_v, tile_n_v, tile_k_v,  (block_q8_0*) src1_col, (int8_t*)src0_row, dst_col);
        }
    }
}

template <int layout_block_n> void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;


    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    GGML_ASSERT(nb11 == (ggml_type_size(src1->type) * ne10));
    GGML_ASSERT(nb12 == (nb11 * ne11));
    GGML_ASSERT(ne12 == 1);
    GGML_ASSERT(ne13 == 1);

    GGML_ASSERT(nb00 == (ggml_type_size(src0->type) * ne00));
    GGML_ASSERT(nb01 == (nb01 * ne01));
    GGML_ASSERT(ne02 == 1);
    GGML_ASSERT(ne03 == 1);
    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    bool is_fp16_type = vec_dot_type == GGML_TYPE_F16;
    bool is_bf16_type = vec_dot_type == GGML_TYPE_BF16;
    bool is_q8_0_type = vec_dot_type == GGML_TYPE_Q8_0;
    bool is_repack = (src0.extra != nullptr) & (src0.op == GGML_OP_NONE);
    GGML_ASSERT(is_fp16_type | is_bf16_type | is_q8_0_type);
    GGML_ASSERT(is_repack);
    
    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);
    const int block_n = layout_block_n;
    // ============================================================================
    // Block-N 对齐的分块策略
    // ============================================================================
    //
    // 目标：确保每个线程处理的数据量是 block_n 的倍数
    // 原因：硬件加速器（如 NFMA）要求数据对齐
    //       - BF16/FP16: block_n = 32  (NFMA = 8)
    //       - FP8:       block_n = 32 (NFMA = 16)
    //       - FP4:       block_n = 32 (NFMA = 32)
    //
    // 策略：
    //   1. chunk_size 向上对齐到 block_n 的倍数
    //   2. 每个线程的工作量向下对齐到 block_n 的倍数
    //   3. 不满足对齐要求的任务分配给空闲线程或跳过

    // This is the size of the first dimension of the result
    const int64_t nr0 = ne0;
    
    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    int chunk_size_n = block_n;
    int chunk_size_m = 8;
    int64_t nchunk0 = (nr0 + chunk_size_n - 1) / chunk_size_n;
    int64_t nchunk1 = (nr1 + chunk_size_m - 1) / chunk_size_m;

    // The number of elements in each chunk
    const int64_t dr0 = chunk_size_n;
    const int64_t dr1 = chunk_size_m;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        ggml_compute_forward_mul_mat_one_chunk<layout_block_n>(params, dst, src0->type, 0, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

}
