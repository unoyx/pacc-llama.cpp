#!/bin/sh

cmake -DCMAKE_SYSTEM_NAME=Linux \
                          -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
                          -DGGML_CPU_RISCV64_LANXIN_MM=ON \
                          -DGGML_PACC_PERF=OFF \
                          -DCMAKE_C_COMPILER=/home/terry/toolchain/linux-22.1/riscv/bin/clang \
                          -DCMAKE_CXX_COMPILER=/home/terry/toolchain/linux-22.1/riscv/bin/clang++ \
                          -DCMAKE_BUILD_TYPE=Release \
                          -DBUILD_SHARED_LIBS=OFF \
                          -DGGML_OPENMP=OFF \
                          -DGGML_RVV=ON \
                          -DGGML_RV_ZFH=ON \
                          -DGGML_RV_ZIHINTPAUSE=OFF \
                          -DGGML_RV_ZVFH=ON \
			  -DGGML_RV_ZVFBFWMA=ON \
                          -DGGML_RV_ZICBOP=ON \
                          -DGGML_VXE=OFF \
                          -S ../pacc-llama.cpp \
                          -G Ninja

