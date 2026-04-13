#pragma once
#include "ggml.h"

static void display_info(const struct ggml_tensor *t)
{
    GGML_LOG_INFO("op type: %s, ", ggml_op_name(t->op));
    GGML_LOG_INFO("dst: %s, ", ggml_get_name(t));

    // GGML_LOG_INFO("data type: %s, ", ggml_type_name(t->type));
    // GGML_LOG_INFO("dim0: %zu, dim1: %zu, dim2: %zu, dim3: %zu, ", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (t->src[i] == NULL) {
            break;
        }
        const struct ggml_tensor *tp = t->src[i];
        GGML_LOG_INFO("src %d, ", i);
        GGML_LOG_INFO(": %s, ", ggml_get_name(tp));
        // GGML_LOG_INFO("data type: %s, ", ggml_type_name(t->type));
        // GGML_LOG_INFO("dim0: %zu, dim1: %zu, dim2: %zu, dim3: %zu, ", tp->ne[0], tp->ne[1], tp->ne[2], tp->ne[3]);
    }
    GGML_LOG_INFO("\n");
}
