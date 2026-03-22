ggml_add_backend_library(ggml-pacc-v0
                         ggml-pacc-v0.cpp)

target_link_libraries(ggml-pacc-v0 PRIVATE m pthread)

if (GGML_OPENMP)
    target_link_libraries(ggml-pacc-v0 PRIVATE OpenMP::OpenMP_CXX)
endif()
