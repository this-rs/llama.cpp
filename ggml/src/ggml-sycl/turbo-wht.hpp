/*
 * TurboQuant Walsh-Hadamard Transform — SYCL backend
 * Port of turbo-wht.cu (CUDA) to SYCL.
 */

#pragma once

#include "common.hpp"

void ggml_sycl_turbo_wht(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
