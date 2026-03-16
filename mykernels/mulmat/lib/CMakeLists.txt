# 收集源文件
file(GLOB LIB_SOURCES
    "src/*.cpp"
)

# 根据架构选择驱动文件
if(TARGET_ARCH STREQUAL "riscv")
    set(ARCH_DEFINES PACC
    )
    set(ARCH_COMPILE_OPTIONS
        -march=rv64gc_zfh_v_zvfh
        -mabi=lp64d
    )
elseif(TARGET_ARCH STREQUAL "x86")
    set(ARCH_DEFINES ""
    )
    set(ARCH_COMPILE_OPTIONS
        -march=native
    )
endif()

file(GLOB LIB_HEADERS
    "include/*.h"
)

# 创建库
add_library(mykernels ${LIB_SOURCES} ${LIB_HEADERS})

# 为库添加头文件路径
target_include_directories(mykernels
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 添加架构特定的编译定义
target_compile_definitions(mykernels
    PUBLIC
        ${ARCH_DEFINES}
    PRIVATE
        GGML_BACKEND_BUILD
        GGML_BACKEND_SHARED
        GGML_SCHED_MAX_COPIES=4
        GGML_SHARED
        GGML_USE_CPU_REPACK
        GGML_USE_LLAMAFILE
        GGML_USE_OPENMP
        _GLIBCXX_ASSERTIONS
        _GNU_SOURCE
        _XOPEN_SOURCE=600
        ggml_cpu_EXPORTS
)

# 添加架构特定的编译选项
target_compile_options(mykernels
    PRIVATE
        ${ARCH_COMPILE_OPTIONS}
        -fPIC
        -Wmissing-declarations
        -Wmissing-noreturn
        -Wall
        -Wextra
        -Wpedantic
        -Wcast-qual
        -Wno-unused-function
        -Wno-array-bounds
        -Wextra-semi
)

# 根据构建类型添加不同的编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(mykernels PRIVATE -g -O0)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(mykernels PRIVATE -O3 -DNDEBUG)
endif()

# 设置库的属性
set_target_properties(mykernels PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# 打印库信息
message(STATUS "Library 'mykernels' created")
