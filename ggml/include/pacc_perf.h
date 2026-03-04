#pragma once

static const char * get_ggml_op_str(const enum ggml_op op) {

    const char * ggml_op_names[GGML_OP_COUNT + 1] = {
        "GGML_OP_NONE",
        "GGML_OP_DUP",
        "GGML_OP_ADD",
        "GGML_OP_ADD_ID",
        "GGML_OP_ADD1",
        "GGML_OP_ACC",
        "GGML_OP_SUB",
        "GGML_OP_MUL",
        "GGML_OP_DIV",
        "GGML_OP_SQR",
        "GGML_OP_SQRT",
        "GGML_OP_LOG",
        "GGML_OP_SIN",
        "GGML_OP_COS",
        "GGML_OP_SUM",
        "GGML_OP_SUM_ROWS",
        "GGML_OP_CUMSUM",
        "GGML_OP_MEAN",
        "GGML_OP_ARGMAX",
        "GGML_OP_COUNT_EQUAL",
        "GGML_OP_REPEAT",
        "GGML_OP_REPEAT_BACK",
        "GGML_OP_CONCAT",
        "GGML_OP_SILU_BACK",
        "GGML_OP_NORM", // normalize
        "GGML_OP_RMS_NORM",
        "GGML_OP_RMS_NORM_BACK",
        "GGML_OP_GROUP_NORM",
        "GGML_OP_L2_NORM",
        "GGML_OP_MUL_MAT",
        "GGML_OP_MUL_MAT_ID",
        "GGML_OP_OUT_PROD",
        "GGML_OP_SCALE",
        "GGML_OP_SET",
        "GGML_OP_CPY",
        "GGML_OP_CONT",
        "GGML_OP_RESHAPE",
        "GGML_OP_VIEW",
        "GGML_OP_PERMUTE",
        "GGML_OP_TRANSPOSE",
        "GGML_OP_GET_ROWS",
        "GGML_OP_GET_ROWS_BACK",
        "GGML_OP_SET_ROWS",
        "GGML_OP_DIAG",
        "GGML_OP_DIAG_MASK_INF",
        "GGML_OP_DIAG_MASK_ZERO",
        "GGML_OP_SOFT_MAX",
        "GGML_OP_SOFT_MAX_BACK",
        "GGML_OP_ROPE",
        "GGML_OP_ROPE_BACK",
        "GGML_OP_CLAMP",
        "GGML_OP_CONV_TRANSPOSE_1D",
        "GGML_OP_IM2COL",
        "GGML_OP_IM2COL_BACK",
        "GGML_OP_IM2COL_3D",
        "GGML_OP_CONV_2D",
        "GGML_OP_CONV_3D",
        "GGML_OP_CONV_2D_DW",
        "GGML_OP_CONV_TRANSPOSE_2D",
        "GGML_OP_POOL_1D",
        "GGML_OP_POOL_2D",
        "GGML_OP_POOL_2D_BACK",
        "GGML_OP_UPSCALE",
        "GGML_OP_PAD",
        "GGML_OP_PAD_REFLECT_1D",
        "GGML_OP_ROLL",
        "GGML_OP_ARANGE",
        "GGML_OP_TIMESTEP_EMBEDDING",
        "GGML_OP_ARGSORT",
        "GGML_OP_TOP_K",
        "GGML_OP_LEAKY_RELU",
        "GGML_OP_TRI",
        "GGML_OP_FILL",
        "GGML_OP_FLASH_ATTN_EXT",
        "GGML_OP_FLASH_ATTN_BACK",
        "GGML_OP_SSM_CONV",
        "GGML_OP_SSM_SCAN",
        "GGML_OP_WIN_PART",
        "GGML_OP_WIN_UNPART",
        "GGML_OP_GET_REL_POS",
        "GGML_OP_ADD_REL_POS",
        "GGML_OP_RWKV_WKV6",
        "GGML_OP_GATED_LINEAR_ATTN",
        "GGML_OP_RWKV_WKV7",
        "GGML_OP_SOLVE_TRI",
        "GGML_OP_UNARY",
        "GGML_OP_MAP_CUSTOM1",
        "GGML_OP_MAP_CUSTOM2",
        "GGML_OP_MAP_CUSTOM3",
        "GGML_OP_CUSTOM",
        "GGML_OP_CROSS_ENTROPY_LOSS",
        "GGML_OP_CROSS_ENTROPY_LOSS_BACK",
        "GGML_OP_OPT_STEP_ADAMW",
        "GGML_OP_OPT_STEP_SGD",
        "GGML_OP_GLU",
        "GGML_OP_COUNT", };

    return ggml_op_names[op];
}

static const char * get_ggml_type_str(const enum ggml_type t) {

    const char * ggml_type_names[GGML_TYPE_COUNT + 1] = {
        "GGML_TYPE_F32",
        "GGML_TYPE_F16",
        "GGML_TYPE_Q4_0",
        "GGML_TYPE_Q4_1",
        "GGML_TYPE_Q4_2", // support has been removed from gguf files
        "GGML_TYPE_Q4_3", // support has been removed from gguf files
        "GGML_TYPE_Q5_0",
        "GGML_TYPE_Q5_1",
        "GGML_TYPE_Q8_0",
        "GGML_TYPE_Q8_1",
        "GGML_TYPE_Q2_K",
        "GGML_TYPE_Q3_K",
        "GGML_TYPE_Q4_K",
        "GGML_TYPE_Q5_K",
        "GGML_TYPE_Q6_K",
        "GGML_TYPE_Q8_K",
        "GGML_TYPE_IQ2_XXS",
        "GGML_TYPE_IQ2_XS",
        "GGML_TYPE_IQ3_XXS",
        "GGML_TYPE_IQ1_S",
        "GGML_TYPE_IQ4_NL",
        "GGML_TYPE_IQ3_S",
        "GGML_TYPE_IQ2_S",
        "GGML_TYPE_IQ4_XS",
        "GGML_TYPE_I8",
        "GGML_TYPE_I16",
        "GGML_TYPE_I32",
        "GGML_TYPE_I64",
        "GGML_TYPE_F64",
        "GGML_TYPE_IQ1_M",
        "GGML_TYPE_BF16",
        "GGML_TYPE_Q4_0_4_4", // support has been removed from gguf files
        "GGML_TYPE_Q4_0_4_8",
        "GGML_TYPE_Q4_0_8_8",
        "GGML_TYPE_TQ1_0",
        "GGML_TYPE_TQ2_0",
        "GGML_TYPE_IQ4_NL_4_4",
        "GGML_TYPE_IQ4_NL_4_8",
        "GGML_TYPE_IQ4_NL_8_8",
        "GGML_TYPE_MXFP4", // MXFP4 (1 block)
        "GGML_TYPE_COUNT",
    };

    return ggml_type_names[t];
}

static void display_info(const struct ggml_tensor *t)
{
    GGML_LOG_INFO("op type: %s, ", get_ggml_op_str(t->op));

    GGML_LOG_INFO("dim0: %zu, dim1: %zu, dim2: %zu, dim3: %zu, ", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (t->src[i] == NULL) {
            break;
        }
        const struct ggml_tensor *tp = t->src[i];
        GGML_LOG_INFO("src %d, ", i);
        GGML_LOG_INFO("dim0: %zu, dim1: %zu, dim2: %zu, dim3: %zu, ", tp->ne[0], tp->ne[1], tp->ne[2], tp->ne[3]);
    }
    GGML_LOG_INFO("\n");
}
