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

# 生成compile_commands.json文件
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    # 添加可执行文件
    add_executable(rv-golden-gen
        src/golden_gen.cpp
        src/tools.cpp
    )

    # 链接库
    target_link_libraries(rv-golden-gen PRIVATE mykernels)

    # 为可执行文件添加头文件路径
    target_include_directories(rv-golden-gen PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(rv-golden-gen PRIVATE
        ARCH_RISCV
    )

    # x86特定编译选项
    target_compile_options(rv-golden-gen PRIVATE
        -march=rv64gc_zfh_v_zvfh
        -mabi=lp64d
    )

    # 添加可执行文件
    add_executable(test-by-golden
        src/test-by-golden.cpp
        src/tools.cpp
    )

    # 链接库
    target_link_libraries(test-by-golden PRIVATE mykernels)

    # 为可执行文件添加头文件路径
    target_include_directories(test-by-golden PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(test-by-golden PRIVATE
        ARCH_RISCV
    )

    # x86特定编译选项
    target_compile_options(test-by-golden PRIVATE
        -march=rv64gc_zfh_v_zvfh
        -mabi=lp64d
    )
elseif(TARGET_ARCH STREQUAL "x86")
    # 添加可执行文件
    add_executable(x86-golden-gen
        src/golden_gen.cpp
        src/tools.cpp
    )

    # 链接库
    target_link_libraries(x86-golden-gen PRIVATE mykernels)

    # 为可执行文件添加头文件路径
    target_include_directories(x86-golden-gen PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(x86-golden-gen PRIVATE
        ARCH_X86
    )

    # x86特定编译选项
    target_compile_options(x86-golden-gen PRIVATE
        -march=native
        -mtune=generic
    )

    # 添加可执行文件
    add_executable(test-by-golden
        src/test-by-golden.cpp
        src/tools.cpp
    )

    # 链接库
    target_link_libraries(test-by-golden PRIVATE mykernels)

    # 为可执行文件添加头文件路径
    target_include_directories(test-by-golden PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(test-by-golden PRIVATE
        ARCH_X86
    )

    # x86特定编译选项
    target_compile_options(test-by-golden PRIVATE
        -march=native
        -mtune=generic
    )
endif()

# 打印构建信息
message(STATUS "Project version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
