#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_pacc_v0_init(void);

GGML_BACKEND_API bool ggml_backend_is_pacc_v0(ggml_backend_t backend);

// number of threads used for zendnn operations
GGML_BACKEND_API void ggml_backend_pacc_v0_set_n_threads(ggml_backend_t backend_pacc_v0, int n_threads);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_pacc_v0_reg(void);

#ifdef __cplusplus
}
#endif
