#pragma once
#include "ggml.h"

static void display_info(const struct ggml_tensor *t)
{
    GGML_LOG_INFO("ternsor name: %s, ", t->name);
    GGML_LOG_INFO("op type: %s, ", ggml_op_name(t->op));
    GGML_LOG_INFO("data type: %s, ", ggml_type_name(t->type));
    GGML_LOG_INFO("ne:[dim0: %zu, dim1: %zu, dim2: %zu, dim3: %zu]", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    GGML_LOG_INFO("nb:[dim0: %zu, dim1: %zu, dim2: %zu, dim3: %zu]", t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
    GGML_LOG_INFO("\n");
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (t->src[i] == NULL) {
            break;
        }
        const struct ggml_tensor *tp = t->src[i];
        GGML_LOG_INFO("  src%d---, ", i);
        GGML_LOG_INFO("ternsor name: %s, ", tp->name);
        GGML_LOG_INFO("op type: %s, ", ggml_op_name(tp->op));
        GGML_LOG_INFO("data type: %s, ", ggml_type_name(tp->type));
        GGML_LOG_INFO("ne:[dim0: %zu, dim1: %zu, dim2: %zu, dim3: %zu]", tp->ne[0], tp->ne[1], tp->ne[2], tp->ne[3]);
        GGML_LOG_INFO("nb:[dim0: %zu, dim1: %zu, dim2: %zu, dim3: %zu]", tp->nb[0], tp->nb[1], tp->nb[2], tp->nb[3]);
        GGML_LOG_INFO("\n");
    }
    GGML_LOG_INFO("\n");
}
		
