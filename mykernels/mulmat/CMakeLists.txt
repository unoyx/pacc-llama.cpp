cmake_minimum_required(VERSION 3.10)
project(MyKernels VERSION 1.0.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置输出路径
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 可以选择构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# 架构选择
set(SUPPORTED_ARCHS "x86" "riscv" CACHE INTERNAL "Supported architectures")
set(TARGET_ARCH "x86" CACHE STRING "Target architecture (x86 or riscv)")

# 验证架构选择
if(NOT TARGET_ARCH IN_LIST SUPPORTED_ARCHS)
    message(FATAL_ERROR "Unsupported architecture: ${TARGET_ARCH}. Supported: ${SUPPORTED_ARCHS}")
endif()

message(STATUS "Building for architecture: ${TARGET_ARCH}")

# 添加库子目录
add_subdirectory(lib)

# 根据架构设置可执行文件的编译选项
if(TARGET_ARCH STREQUAL "riscv")
    target_compile_definitions(myapp PRIVATE
        ARCH_RISCV
        RISCV_VERSION=1
    )

    # RISC-V特定编译选项
    target_compile_options(myapp PRIVATE
        -march=rv64gc
        -mabi=lp64d
    )

    # RISC-V特定链接选项
    target_link_options(myapp PRIVATE
        -static
    )
elseif(TARGET_ARCH STREQUAL "x86")
    # 添加可执行文件
    add_executable(x86-gemm-driver
        src/gemm-driver.cpp
    )

    # 链接库
    target_link_libraries(x86-gemm-driver PRIVATE mykernels)

    # 为可执行文件添加头文件路径
    target_include_directories(x86-gemm-driver PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(x86-gemm-driver PRIVATE
        ARCH_X86
    )

    # x86特定编译选项
    target_compile_options(x86-gemm-driver PRIVATE
        -march=native
        -mtune=generic
    )
endif()

# 打印构建信息
message(STATUS "Project version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
