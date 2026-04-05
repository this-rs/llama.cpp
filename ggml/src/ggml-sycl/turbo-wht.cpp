/*
 * TurboQuant Walsh-Hadamard Transform — SYCL kernel
 * Port of turbo-wht.cu (CUDA) to SYCL.
 *
 * One workgroup of 128 threads = one 128-element WHT group.
 * Forward WHT:  signs1 → butterfly (7 stages) → signs2 → normalize
 * Inverse WHT:  signs2 → butterfly (7 stages) → signs1 → normalize
 *
 * The sign arrays and normalization constant must match turbo-quant.cuh EXACTLY.
 * InnerQ channel scaling is NOT yet implemented (TODO for later).
 */

#include "turbo-wht.hpp"
#include "turbo-quant.hpp"

// ─── SYCL kernel ─────────────────────────────────────────────────────────────

template <int direction>
static void k_turbo_wht_f32_sycl(const float * __restrict__ src,
                                  float * __restrict__ dst,
                                  int64_t n_groups,
                                  int64_t head_dim,
                                  int64_t groups_per_head,
                                  const sycl::nd_item<3> & item_ct1,
                                  float * smem) {
    const int64_t g = item_ct1.get_group(2);
    if (g >= n_groups) return;

    const int t = item_ct1.get_local_id(2); // 0..127

    // Map group index → position in tensor
    const int64_t head_idx     = g / groups_per_head;
    const int64_t grp_in_head  = g % groups_per_head;
    const int64_t base         = head_idx * head_dim + grp_in_head * 128;

    // Load from global memory
    smem[t] = src[base + t];
    sycl::group_barrier(item_ct1.get_group());

    // Apply first sign array
    smem[t] *= (direction == 0) ? TURBO_WHT_SIGNS1_SYCL[t] : TURBO_WHT_SIGNS2_SYCL[t];
    sycl::group_barrier(item_ct1.get_group());

    // WHT butterfly — 7 stages (h = 1, 2, 4, 8, 16, 32, 64)
#define WHT_STAGE_SYCL(h) \
    if ((t % (2*(h))) < (h)) { \
        float a = smem[t], b = smem[t+(h)]; \
        smem[t] = a + b; smem[t+(h)] = a - b; \
    } \
    sycl::group_barrier(item_ct1.get_group());

    WHT_STAGE_SYCL(1)
    WHT_STAGE_SYCL(2)
    WHT_STAGE_SYCL(4)
    WHT_STAGE_SYCL(8)
    WHT_STAGE_SYCL(16)
    WHT_STAGE_SYCL(32)
    WHT_STAGE_SYCL(64)
#undef WHT_STAGE_SYCL

    // Normalize and apply second sign array, write to output
    float result = smem[t] * TURBO_INV_SQRT_128_SYCL
                 * ((direction == 0) ? TURBO_WHT_SIGNS2_SYCL[t] : TURBO_WHT_SIGNS1_SYCL[t]);

    dst[base + t] = result;
}

// ─── Simple copy kernel for tail elements (identity pass-through) ────────────

static void k_turbo_wht_copy_tail_sycl(const float * __restrict__ src,
                                         float * __restrict__ dst,
                                         int64_t n_heads,
                                         int64_t head_dim,
                                         int64_t tail_offset,
                                         int tail_size,
                                         const sycl::nd_item<3> & item_ct1) {
    const int64_t i = (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if (i >= n_heads * tail_size) return;

    const int64_t head_idx  = i / tail_size;
    const int64_t tail_elem = i % tail_size;
    const int64_t offset    = head_idx * head_dim + tail_offset + tail_elem;
    dst[offset] = src[offset];
}

// ─── Dispatch ─────────────────────────────────────────────────────────────────

void ggml_sycl_turbo_wht(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int direction;
    int group_size;
    memcpy(&direction,  dst->op_params + 0, sizeof(int));
    memcpy(&group_size, dst->op_params + sizeof(int), sizeof(int));

    const int64_t head_dim        = src->ne[0];
    const int64_t n_heads         = ggml_nelements(src) / head_dim;

    // Only 128-element groups supported in SYCL for now
    // (64-element groups are rare and can be added later)
    GGML_ASSERT(group_size == 128);
    const int64_t groups_per_head = head_dim / group_size;
    const int     tail_size       = (int)(head_dim % group_size);
    const int64_t n_groups        = groups_per_head * n_heads;

    const float * src_ptr = (const float *) src->data;
    float       * dst_ptr = (float       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    // Process full groups
    if (n_groups > 0) {
        if (direction == 0) {
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> smem_acc(sycl::range<1>(128), cgh);
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) * sycl::range<3>(1, 1, 128),
                                      sycl::range<3>(1, 1, 128)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_turbo_wht_f32_sycl<0>(src_ptr, dst_ptr, n_groups, head_dim,
                                                 groups_per_head, item_ct1,
                                                 smem_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
        } else {
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> smem_acc(sycl::range<1>(128), cgh);
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) * sycl::range<3>(1, 1, 128),
                                      sycl::range<3>(1, 1, 128)),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_turbo_wht_f32_sycl<1>(src_ptr, dst_ptr, n_groups, head_dim,
                                                 groups_per_head, item_ct1,
                                                 smem_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
        }
    }

    // Pass through tail elements unchanged (no rotation)
    if (tail_size > 0) {
        const int64_t total_tail = n_heads * tail_size;
        const int block_sz = 256;
        const int64_t n_blocks = (total_tail + block_sz - 1) / block_sz;
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, n_blocks) * sycl::range<3>(1, 1, block_sz),
                              sycl::range<3>(1, 1, block_sz)),
            [=](sycl::nd_item<3> item_ct1) {
                k_turbo_wht_copy_tail_sycl(src_ptr, dst_ptr, n_heads, head_dim,
                                            groups_per_head * 128, tail_size, item_ct1);
            });
    }
}
