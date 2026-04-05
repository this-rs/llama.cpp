/*
 * TurboQuant constants and helpers for SYCL backend
 * Port of turbo-quant.cuh (CUDA) to SYCL.
 *
 * Constants must match turbo-quant.cuh EXACTLY — a single bit difference
 * produces garbage output (learned the hard way on Vulkan, commit 4457182c0).
 */

#pragma once

#include "common.hpp"

// ---- Quantization ratio for dequantize_block template ----
// Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TURBO3 1

// ---- 3-bit centroids (Lloyd-Max for N(0, 1/128)) ----
// Must match turbo-quant.cuh TURBO_CENTROIDS_3BIT EXACTLY
static constexpr float TURBO_CENTROIDS_3BIT_SYCL[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// ---- Midpoints for nearest centroid lookup ----
// Must match turbo-quant.cuh TURBO_MID_3BIT EXACTLY
static constexpr float TURBO_MID_3BIT_SYCL[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// ---- WHT sign arrays (seed=42) ----
// Must match turbo-quant.cuh TURBO_WHT_SIGNS1/SIGNS2 EXACTLY

static constexpr float TURBO_WHT_SIGNS1_SYCL[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static constexpr float TURBO_WHT_SIGNS2_SYCL[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

static constexpr float TURBO_INV_SQRT_128_SYCL = 0.08838834764831845f;

// ---- Dequantize a single element from a turbo3 block ----
// j = element index within block (0..127)
// norm = float value of block norm
static __dpct_inline__ float turbo3_dequant_element_sycl(
        const block_turbo3_0 * __restrict__ x, int j, float norm) {
    const uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    const uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    const uint8_t idx  = low2 | (hi1 << 2);
    return TURBO_CENTROIDS_3BIT_SYCL[idx] * norm;
}

// ---- Nearest 3-bit centroid via binary search on midpoints ----
// Returns index 0..7 of nearest centroid for a normalized value
static __dpct_inline__ uint8_t turbo_nearest_centroid_3bit_sycl(float v) {
    if (v > TURBO_MID_3BIT_SYCL[3]) {
        return v > TURBO_MID_3BIT_SYCL[5]
            ? (v > TURBO_MID_3BIT_SYCL[6] ? 7 : 6)
            : (v > TURBO_MID_3BIT_SYCL[4] ? 5 : 4);
    } else {
        return v > TURBO_MID_3BIT_SYCL[1]
            ? (v > TURBO_MID_3BIT_SYCL[2] ? 3 : 2)
            : (v > TURBO_MID_3BIT_SYCL[0] ? 1 : 0);
    }
}
