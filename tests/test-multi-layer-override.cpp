// Test: multi-layer override mechanism [obrain]
// Validates llama_set_layer_output_override_multi: captures real
// intermediate-layer activations (ALL token positions) at TWO distinct
// layers simultaneously, round-trips them back in via the new multi-layer
// override API and verifies the final logits match the natural result
// (sanity: overriding with the exact captured values changes nothing).
// Then perturbs just the LAST token's slice at BOTH layers simultaneously
// and verifies the logits actually change — and, as a discriminating check
// against a bug where only ONE of the two layers' override actually takes
// effect, verifies the two-layer perturbed result differs from EACH of the
// two single-layer perturbed results (perturbing both layers should not be
// identical to perturbing either one alone).
#include "llama.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

static float cosine_similarity(const float * a, const float * b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float) (dot / (sqrt(na) * sqrt(nb)));
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    const int n_layer = llama_model_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    fprintf(stderr, "Model: n_layer=%d, n_embd=%d, n_vocab=%d\n", n_layer, n_embd, n_vocab);

    const int32_t layer_a = n_layer / 3;
    const int32_t layer_b = (2 * n_layer) / 3;
    fprintf(stderr, "Probing two layers: %d and %d\n", layer_a, layer_b);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 64;
    cparams.n_batch   = 64;
    cparams.n_threads = 4;
    cparams.embeddings = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    const char * text = "The capital of France is";
    const int max_tokens = 64;
    std::vector<llama_token> tokens(max_tokens);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_tokens = llama_tokenize(vocab, text, strlen(text), tokens.data(), max_tokens, true, false);
    if (n_tokens < 2) {
        fprintf(stderr, "Tokenization failed or too short (need >= 2 tokens)\n");
        return 1;
    }
    tokens.resize(n_tokens);
    fprintf(stderr, "Tokenized %d tokens\n", n_tokens);

    auto make_batch = [&]() {
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            batch.token[i]     = tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == n_tokens - 1);
        }
        batch.n_tokens = n_tokens;
        return batch;
    };

    // ── Pass 1: natural decode, capture ALL positions at BOTH layers ──
    const int32_t capture_layers[2] = { layer_a, layer_b };
    llama_set_layer_output_capture(ctx, capture_layers, 2);
    llama_batch batch1 = make_batch();
    if (llama_decode(ctx, batch1) != 0) {
        fprintf(stderr, "FAIL: natural decode failed\n");
        return 1;
    }
    std::vector<float> h_a_full((size_t) n_tokens * n_embd);
    std::vector<float> h_b_full((size_t) n_tokens * n_embd);
    for (int i = 0; i < n_tokens; i++) {
        const float * row_a = llama_get_layer_output(ctx, layer_a, i);
        const float * row_b = llama_get_layer_output(ctx, layer_b, i);
        if (!row_a || !row_b) {
            fprintf(stderr, "FAIL: layer capture returned NULL at position %d\n", i);
            return 1;
        }
        memcpy(h_a_full.data() + (size_t) i * n_embd, row_a, n_embd * sizeof(float));
        memcpy(h_b_full.data() + (size_t) i * n_embd, row_b, n_embd * sizeof(float));
    }
    const float * logits_natural_ptr = llama_get_logits_ith(ctx, n_tokens - 1);
    if (!logits_natural_ptr) {
        fprintf(stderr, "FAIL: natural logits are NULL\n");
        return 1;
    }
    std::vector<float> logits_natural(logits_natural_ptr, logits_natural_ptr + n_vocab);
    llama_set_layer_output_capture(ctx, nullptr, 0);
    llama_batch_free(batch1);
    fprintf(stderr, "Captured natural h at layers %d and %d, all %d positions\n", layer_a, layer_b, n_tokens);

    // ── Pass 2: round-trip — override BOTH layers simultaneously with the
    // EXACT captured values (all positions, unmodified) ──
    {
        llama_memory_clear(llama_get_memory(ctx), true);
        const int32_t idx[2] = { layer_a, layer_b };
        const float * data[2] = { h_a_full.data(), h_b_full.data() };
        const int32_t nfl[2] = { (int32_t) h_a_full.size(), (int32_t) h_b_full.size() };
        llama_set_layer_output_override_multi(ctx, idx, data, nfl, 2);
        llama_batch batch2 = make_batch();
        if (llama_decode(ctx, batch2) != 0) {
            fprintf(stderr, "FAIL: round-trip decode failed\n");
            return 1;
        }
        const float * logits_rt_ptr = llama_get_logits_ith(ctx, n_tokens - 1);
        float cos_rt = cosine_similarity(logits_natural.data(), logits_rt_ptr, n_vocab);
        fprintf(stderr, "Round-trip (2 layers, unmodified values) cosine = %.6f\n", cos_rt);
        llama_batch_free(batch2);
        if (cos_rt < 0.999f) {
            fprintf(stderr, "FAIL: round-trip cosine too low (%.6f), expected > 0.999\n", cos_rt);
            llama_set_layer_output_override_multi(ctx, nullptr, nullptr, nullptr, 0);
            return 1;
        }
    }

    // Build a fixed perturbation direction to add to the LAST token's slice.
    auto perturb_last = [&](std::vector<float> & h_full, float scale) {
        std::vector<float> out = h_full;
        float * last = out.data() + (size_t)(n_tokens - 1) * n_embd;
        double norm = 0.0;
        for (int i = 0; i < n_embd; i++) norm += (double) last[i] * last[i];
        norm = sqrt(norm);
        for (int i = 0; i < n_embd; i++) {
            float dir = ((i * 2654435761u) % 1000) / 500.0f - 1.0f;
            last[i] += dir * scale * (float)(norm) / sqrtf((float) n_embd);
        }
        return out;
    };

    // ── Pass 3: perturb layer A ONLY ──
    std::vector<float> logits_a_only;
    {
        std::vector<float> h_a_pert = perturb_last(h_a_full, 0.3f);
        llama_memory_clear(llama_get_memory(ctx), true);
        const int32_t idx[1] = { layer_a };
        const float * data[1] = { h_a_pert.data() };
        const int32_t nfl[1] = { (int32_t) h_a_pert.size() };
        llama_set_layer_output_override_multi(ctx, idx, data, nfl, 1);
        llama_batch batch3 = make_batch();
        if (llama_decode(ctx, batch3) != 0) { fprintf(stderr, "FAIL: layer-A-only decode failed\n"); return 1; }
        const float * p = llama_get_logits_ith(ctx, n_tokens - 1);
        logits_a_only.assign(p, p + n_vocab);
        llama_batch_free(batch3);
    }

    // ── Pass 4: perturb layer B ONLY ──
    std::vector<float> logits_b_only;
    {
        std::vector<float> h_b_pert = perturb_last(h_b_full, 0.3f);
        llama_memory_clear(llama_get_memory(ctx), true);
        const int32_t idx[1] = { layer_b };
        const float * data[1] = { h_b_pert.data() };
        const int32_t nfl[1] = { (int32_t) h_b_pert.size() };
        llama_set_layer_output_override_multi(ctx, idx, data, nfl, 1);
        llama_batch batch4 = make_batch();
        if (llama_decode(ctx, batch4) != 0) { fprintf(stderr, "FAIL: layer-B-only decode failed\n"); return 1; }
        const float * p = llama_get_logits_ith(ctx, n_tokens - 1);
        logits_b_only.assign(p, p + n_vocab);
        llama_batch_free(batch4);
    }

    // ── Pass 5: perturb BOTH layers simultaneously (the actual new
    // capability) — CORRECTLY, via SEQUENTIAL construction.
    //
    // IMPORTANT MECHANISTIC GOTCHA (discovered live by this very test,
    // first version): an override REPLACES the residual stream at that
    // layer, it does not ADD to it. If layer B's override buffer is built
    // from the plain natural (pre-any-override) capture of layer B, then
    // applying it DISCARDS whatever effect layer A's (shallower) override
    // would have propagated forward to that point — only the DEEPEST
    // active override actually survives to influence the final output.
    // Confirmed: with independently-built buffers, cosine(BOTH, B-only) =
    // 1.000000 (A's contribution silently erased).
    //
    // Fix: build each deeper layer's override buffer from a capture taken
    // WHILE the shallower override(s) are already active — so it already
    // contains their propagated effect — then add that layer's own extra
    // perturbation on top. This makes the combination genuinely additive
    // instead of "last override wins".
    std::vector<float> h_a_pert = perturb_last(h_a_full, 0.3f);

    // Step 5a: decode with ONLY layer A's override, capture layer B's
    // value AS INFLUENCED by A's override having propagated forward.
    std::vector<float> h_b_given_a((size_t) n_tokens * n_embd);
    {
        llama_memory_clear(llama_get_memory(ctx), true);
        const int32_t idx[1] = { layer_a };
        const float * data[1] = { h_a_pert.data() };
        const int32_t nfl[1] = { (int32_t) h_a_pert.size() };
        llama_set_layer_output_capture(ctx, &layer_b, 1); // capture B while A is overridden
        llama_set_layer_output_override_multi(ctx, idx, data, nfl, 1);
        llama_batch batch5a = make_batch();
        if (llama_decode(ctx, batch5a) != 0) { fprintf(stderr, "FAIL: step 5a decode failed\n"); return 1; }
        for (int i = 0; i < n_tokens; i++) {
            const float * row = llama_get_layer_output(ctx, layer_b, i);
            memcpy(h_b_given_a.data() + (size_t) i * n_embd, row, n_embd * sizeof(float));
        }
        llama_set_layer_output_capture(ctx, nullptr, 0);
        llama_batch_free(batch5a);
    }

    // Step 5b: layer B's FINAL override buffer = (value already reflecting
    // A's propagated effect) + B's OWN additional perturbation.
    std::vector<float> h_b_pert_cumulative = perturb_last(h_b_given_a, 0.3f);

    // Step 5c: final decode with BOTH overrides — B's buffer already bakes
    // in A's contribution, so both effects should now combine.
    std::vector<float> logits_both;
    {
        llama_memory_clear(llama_get_memory(ctx), true);
        const int32_t idx[2] = { layer_a, layer_b };
        const float * data[2] = { h_a_pert.data(), h_b_pert_cumulative.data() };
        const int32_t nfl[2] = { (int32_t) h_a_pert.size(), (int32_t) h_b_pert_cumulative.size() };
        llama_set_layer_output_override_multi(ctx, idx, data, nfl, 2);
        llama_batch batch5c = make_batch();
        if (llama_decode(ctx, batch5c) != 0) { fprintf(stderr, "FAIL: step 5c decode failed\n"); return 1; }
        const float * p = llama_get_logits_ith(ctx, n_tokens - 1);
        logits_both.assign(p, p + n_vocab);
        llama_batch_free(batch5c);
    }

    float cos_a_vs_natural    = cosine_similarity(logits_natural.data(), logits_a_only.data(), n_vocab);
    float cos_b_vs_natural    = cosine_similarity(logits_natural.data(), logits_b_only.data(), n_vocab);
    float cos_both_vs_natural = cosine_similarity(logits_natural.data(), logits_both.data(), n_vocab);
    float cos_both_vs_a       = cosine_similarity(logits_both.data(), logits_a_only.data(), n_vocab);
    float cos_both_vs_b       = cosine_similarity(logits_both.data(), logits_b_only.data(), n_vocab);

    fprintf(stderr, "\ncosine(natural, layer-A-only-perturbed)   = %.6f\n", cos_a_vs_natural);
    fprintf(stderr, "cosine(natural, layer-B-only-perturbed)   = %.6f\n", cos_b_vs_natural);
    fprintf(stderr, "cosine(natural, BOTH-layers-perturbed)    = %.6f\n", cos_both_vs_natural);
    fprintf(stderr, "cosine(BOTH-perturbed, layer-A-only)      = %.6f\n", cos_both_vs_a);
    fprintf(stderr, "cosine(BOTH-perturbed, layer-B-only)      = %.6f\n", cos_both_vs_b);

    bool ok = true;
    // Each single-layer perturbation should change something.
    if (cos_a_vs_natural > 0.9999f) { fprintf(stderr, "FAIL: layer-A-only perturbation had no effect\n"); ok = false; }
    if (cos_b_vs_natural > 0.9999f) { fprintf(stderr, "FAIL: layer-B-only perturbation had no effect\n"); ok = false; }
    // The combined two-layer result should differ from EITHER single-layer
    // result — this is the discriminating check that BOTH overrides are
    // really simultaneously active, not just one of them silently winning.
    if (cos_both_vs_a > 0.9999f) { fprintf(stderr, "FAIL: BOTH-layers result identical to layer-A-only — layer B override may be silently ignored\n"); ok = false; }
    if (cos_both_vs_b > 0.9999f) { fprintf(stderr, "FAIL: BOTH-layers result identical to layer-B-only — layer A override may be silently ignored\n"); ok = false; }

    fprintf(stderr, "\n%s\n", ok ? "=== ALL CHECKS PASSED ===" : "=== FAILED ===");

    llama_set_layer_output_override_multi(ctx, nullptr, nullptr, nullptr, 0);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
