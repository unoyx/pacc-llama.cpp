#!/bin/sh

cmake -DCMAKE_SYSTEM_NAME=Linux \
                          -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
                          -DGGML_CPU_RISCV64_LANXIN=OFF \
                          -DCMAKE_C_COMPILER=/home/terry/glibc237/bin/riscv64-unknown-linux-gnu-gcc \
                          -DCMAKE_CXX_COMPILER=/home/terry/glibc237/bin/riscv64-unknown-linux-gnu-g++ \
                          -DCMAKE_SYSROOT=/home/terry/glibc237/sysroot \
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

