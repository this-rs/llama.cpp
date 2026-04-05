#include "set_rows.hpp"
#include "cpy.hpp"
#include "turbo-quant.hpp"

namespace utils {
template<typename T>
static constexpr bool is_arithmetic_v() {
    return std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half> || std::is_same_v<T, sycl::ext::oneapi::bfloat16>;
}
}

template<typename TIn, typename TOut>
static inline std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), void>
convert (const char* src, char* dst) {
    auto src_val = *reinterpret_cast<const TIn*>(src);
    auto dst_val = sycl::vec<TIn, 1>(src_val).template convert<TOut, sycl::rounding_mode::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

template <typename TIdx, typename blockType, int qk, cpy_kernel_t cpyblck>
static void set_rows_sycl_q(const char * __restrict__ src0_d,
                            const TIdx * __restrict__ src1_d,
                            blockType * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            const int64_t ne00,
                            const int64_t ne01,
                            const int64_t ne02,
                            const int64_t ne03,
                            const int64_t ne10,
                            const int64_t ne11,
                            const int64_t ne12,
                            const int64_t ne13,
                            // strides for src0
                            const size_t  nb00,
                            const size_t  nb01,
                            const size_t  nb02,
                            const size_t  nb03,
                            // strides for src1
                            const size_t  nb10,
                            const size_t  nb11,
                            const size_t  nb12,
                            const size_t  nb13,
                            // strides for dst
                            const size_t  nb1,
                            const size_t  nb2,
                            const size_t  nb3,
                            queue_ptr     stream) {
    const int64_t total_blocks = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr int block_size   = 256;
    const int64_t grid_size    = ceil_div(total_blocks, block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item_ct1) {
        const int64_t i = item_ct1.get_global_linear_id();
        if (i >= total_blocks) {
            return;
        }
        const int64_t i_base      = i * qk;
        const int64_t i03         = i_base / (ne00 * ne01 * ne02);
        const int64_t rem1        = i_base - i03 * (ne00 * ne01 * ne02);
        const int64_t i02         = rem1 / (ne00 * ne01);
        const int64_t rem2        = rem1 - i02 * ne00 * ne01;
        const int64_t i01         = rem2 / ne00;
        const int64_t i00         = rem2 - i01 * ne00;
        const int64_t i12         = i03 % ne12;
        const int64_t i11         = i02 % ne11;
        const int64_t i10         = i01;
        const size_t  src_offset  = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        const char *  src_block   = src0_d + src_offset + i00 * sizeof(float);
        const size_t  src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        const int64_t dst_row     = src1_d[src1_offset / sizeof(TIdx)];
        const size_t  dst_offset =
            calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 }) + (i00 / qk) * sizeof(blockType);
        char * dst_block = reinterpret_cast<char *>(reinterpret_cast<char *>(dst_d) + dst_offset);
        cpyblck(src_block, dst_block);
    });
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(nb00);
    GGML_UNUSED(nb13);
}

template<typename TIn, typename TIdx, typename TOut>
static void k_set_rows(
        const char * __restrict__ src0, const TIdx * __restrict__ src1, char * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne11, const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size,
        const int64_t total_elements,
        const sycl::nd_item<1> & item_ct1) {

    const int64_t i = item_ct1.get_global_linear_id();
    if (i >= total_elements) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(const TIdx *)((const char *)src1 + calculate_offset<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    const char * src0_row = src0 + calculate_offset<3>({nb01, nb02, nb03}, {i01, i02, i03});
    const char * src_elem = src0_row + i00 * src_type_size;
    char * dst_row_ptr = dst + dst_row*nb1 + i02*nb2 + i03*nb3;
    char * dst_elem = dst_row_ptr + i00 * dst_type_size;

    convert<TIn, TOut>(src_elem, dst_elem);
}

template<typename TIn, typename TIdx, typename TOut>
static void set_rows_sycl(
        const char * src0_d, const TIdx * src1_d, char * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne11, const int64_t ne12, const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t src_type_size, const size_t dst_type_size,
        queue_ptr stream) {

    const int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    constexpr int block_size = 64;
    const int64_t grid_size = ceil_div(total_elements, block_size);

    stream->parallel_for(
        sycl::nd_range<1>(grid_size * block_size, block_size),
        [=](sycl::nd_item<1> item_ct1) {
            k_set_rows<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_size, dst_type_size,
                total_elements,
                item_ct1
            );
        }
    );
}

// ─── TurboQuant3 set_rows: cooperative 128-thread quantization kernel ────────
//
// One workgroup of 128 threads = one 128-element WHT group.
// Pipeline: load → L2 norm → normalize → forward WHT → quantize 3-bit
//           → pack qs[32]/signs[16] → reconstruction norm → corrected norm
//
// CUDA uses __shfl_xor_sync / __ballot_sync for reductions and packing.
// SYCL port uses shared-memory tree reductions and cooperative gather,
// which avoids any sub-group size assumption (works with sg=16 or sg=32).
//
// InnerQ channel scaling is NOT yet implemented (TODO for later).

template <typename idx_t>
static void k_set_rows_turbo3_sycl(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        char * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3,
        const int64_t n_groups_per_row,
        const sycl::nd_item<3> & item_ct1,
        float * smem,
        float * red,
        uint8_t * sidx) {

    const int j = item_ct1.get_local_id(2);   // 0..127
    const int64_t g = item_ct1.get_group(2);   // flat group index

    // Decode g → (i_grp, i01, i02, i03)
    const int64_t i_grp = g % n_groups_per_row;
    int64_t tmp = g / n_groups_per_row;
    const int64_t i01 = tmp % ne01;
    tmp = tmp / ne01;
    const int64_t i02 = tmp % ne12;
    const int64_t i03 = tmp / ne12;

    const int64_t i10 = i01;
    const int64_t i11_val = i01 % ne11;
    const int64_t i12 = i02;

    const int64_t dst_row = *(src1 + i10*s10 + i11_val*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    // blocks_per_group = 128 / QK_TURBO3 = 1, so one block per group
    block_turbo3_0 * blk = (block_turbo3_0 *)(dst + dst_row*nb1 + i02*nb2 + i03*nb3) + i_grp;

    // ── Step 1: Load element j (coalesced) ──
    smem[j] = src_row[i_grp * 128 + j];
    sycl::group_barrier(item_ct1.get_group());

    // ── Step 2: L2 norm (tree reduction in shared memory) ──
    red[j] = smem[j] * smem[j];
    sycl::group_barrier(item_ct1.get_group());

    for (int stride = 64; stride > 0; stride >>= 1) {
        if (j < stride) red[j] += red[j + stride];
        sycl::group_barrier(item_ct1.get_group());
    }
    const float grp_norm = sycl::sqrt(red[0]);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ── Step 3: Normalize ──
    smem[j] *= inv_norm;
    sycl::group_barrier(item_ct1.get_group());

    // ── Step 4: Forward WHT (signs1 → butterfly → signs2 × INV_SQRT_128) ──
    smem[j] *= TURBO_WHT_SIGNS1_SYCL[j];
    sycl::group_barrier(item_ct1.get_group());

#define WHT_STAGE_SR(h) \
    if ((j % (2*(h))) < (h)) { \
        float a = smem[j], b = smem[j+(h)]; \
        smem[j] = a + b; smem[j+(h)] = a - b; \
    } \
    sycl::group_barrier(item_ct1.get_group());

    WHT_STAGE_SR(1)
    WHT_STAGE_SR(2)
    WHT_STAGE_SR(4)
    WHT_STAGE_SR(8)
    WHT_STAGE_SR(16)
    WHT_STAGE_SR(32)
    WHT_STAGE_SR(64)
#undef WHT_STAGE_SR

    smem[j] = smem[j] * TURBO_INV_SQRT_128_SYCL * TURBO_WHT_SIGNS2_SYCL[j];
    sycl::group_barrier(item_ct1.get_group());

    // ── Step 5: Quantize element j to 3-bit centroid index ──
    const uint8_t idx = turbo_nearest_centroid_3bit_sycl(smem[j]);

    // ── Step 6: Pack qs and signs (shared-memory cooperative, no warp intrinsics) ──
    sidx[j] = idx;
    sycl::group_barrier(item_ct1.get_group());

    // Pack qs: 4 elements → 1 byte (32 qs bytes per turbo3 block)
    if (j < 32) {
        const int base = j * 4;
        blk->qs[j] = (sidx[base] & 0x3)
                    | ((sidx[base + 1] & 0x3) << 2)
                    | ((sidx[base + 2] & 0x3) << 4)
                    | ((sidx[base + 3] & 0x3) << 6);
    }

    // Pack signs: 8 elements → 1 byte (16 signs bytes per turbo3 block)
    if (j < 16) {
        const int base = j * 8;
        uint8_t byte = 0;
        for (int k = 0; k < 8; k++) {
            byte |= ((sidx[base + k] >> 2) & 0x1) << k;
        }
        blk->signs[j] = byte;
    }

    // ── Step 7: Reconstruction norm (same tree reduction pattern) ──
    const float c = TURBO_CENTROIDS_3BIT_SYCL[idx];
    red[j] = c * c;
    sycl::group_barrier(item_ct1.get_group());

    for (int stride = 64; stride > 0; stride >>= 1) {
        if (j < stride) red[j] += red[j + stride];
        sycl::group_barrier(item_ct1.get_group());
    }
    const float recon_norm     = sycl::sqrt(red[0]);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    // ── Step 8: Write corrected norm (one thread per block) ──
    // NOTE: blk->norm is sycl::half (= ggml_half in SYCL context).
    // Do NOT use GGML_FP32_TO_FP16 here — it returns uint16_t (ggml_fp16_t),
    // and assigning uint16_t to sycl::half does integer value conversion
    // (e.g. 15360 → 15360.0) instead of bit reinterpretation (0x3C00 → 1.0).
    // Simple cast float→sycl::half is the correct GPU-side conversion.
    if (j == 0) {
        blk->norm = sycl::half(corrected_norm);
    }
}

// ─── Dispatch wrapper for turbo3 set_rows ────────────────────────────────────

template <typename idx_t>
static void set_rows_turbo3_sycl(ggml_backend_sycl_context & ctx,
                                  const ggml_tensor * src0,
                                  const ggml_tensor * src1,
                                  ggml_tensor * dst) {
    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;
    char        * dst_d  = (char *)dst->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    // Only 128-element groups supported in SYCL for now
    int group_size = 128;
    memcpy(&group_size, dst->op_params, sizeof(int));
    if (group_size != 128) group_size = 128;
    GGML_ASSERT(ne00 % group_size == 0);
    GGML_ASSERT(ne00 % QK_TURBO3 == 0);

    dpct::queue_ptr stream = ctx.stream();

    const int64_t n_groups_per_row = ne00 / group_size;
    const int64_t n_total_groups   = n_groups_per_row * ne01 * ne02 * ne03;

    // Convert byte strides to element strides for src0 (float) and src1 (idx_t)
    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    if (n_total_groups > 0) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1>   smem_acc(sycl::range<1>(128), cgh);
            sycl::local_accessor<float, 1>   red_acc(sycl::range<1>(128), cgh);
            sycl::local_accessor<uint8_t, 1> sidx_acc(sycl::range<1>(128), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, n_total_groups) * sycl::range<3>(1, 1, 128),
                    sycl::range<3>(1, 1, 128)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_set_rows_turbo3_sycl<idx_t>(
                        src0_d, src1_d, dst_d,
                        ne00, ne01, ne11, ne12,
                        s01, s02, s03, s10, s11, s12,
                        (int64_t)nb1, (int64_t)nb2, (int64_t)nb3,
                        n_groups_per_row,
                        item_ct1,
                        smem_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        red_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        sidx_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

template<typename TIn, typename TIdx>
static void set_rows_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const char * src0_d = (const char *)src0->data;
    const TIdx * src1_d = (const TIdx *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    dpct::queue_ptr stream = ctx.stream();
    switch (dst->type) {
        case GGML_TYPE_F32:
            set_rows_sycl<TIn, TIdx, float>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(TIn), sizeof(float),
                stream
            );
            break;
        case GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            set_rows_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(TIn), sizeof(sycl::half),
                stream
            );
            break;
        case GGML_TYPE_BF16:
            set_rows_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (char *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                sizeof(TIn), sizeof(sycl::ext::oneapi::bfloat16),
                stream
            );
            break;
        case GGML_TYPE_Q8_0:
            set_rows_sycl_q<TIdx, block_q8_0, QK8_0, cpy_blck_f32_q8_0>(src0_d, src1_d, (block_q8_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            set_rows_sycl_q<TIdx, block_q5_1, QK5_1, cpy_blck_f32_q5_1>(src0_d, src1_d, (block_q5_1 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            set_rows_sycl_q<TIdx, block_q5_0, QK5_0, cpy_blck_f32_q5_0>(src0_d, src1_d, (block_q5_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            set_rows_sycl_q<TIdx, block_q4_1, QK4_1, cpy_blck_f32_q4_1>(src0_d, src1_d, (block_q4_1 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            set_rows_sycl_q<TIdx, block_q4_0, QK4_0, cpy_blck_f32_q4_0>(src0_d, src1_d, (block_q4_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            set_rows_sycl_q<TIdx, block_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(src0_d, src1_d, (block_iq4_nl *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_TURBO3_0:
            set_rows_turbo3_sycl<TIdx>(ctx, src0, src1, dst);
            break;

        default:
            GGML_ABORT("Unsupported tensor type!");
            break;
    }
}

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_sycl<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_sycl<float, int32_t>(ctx, src0, src1, dst);
    }
}
