// Test: H1 multi-layer — layer output override mechanism [obrain]
// Validates llama_set_layer_output_override: captures a real intermediate
// layer activation (ALL token positions) via the existing capture
// mechanism, then feeds those SAME values back in via the new override
// mechanism at the SAME layer and verifies the resulting final logits
// match the natural (un-overridden) result exactly (round-trip sanity
// check). Then perturbs just the LAST token's slice and verifies the final
// logits actually change (sanity that the override has real effect, not
// silently ignored).
//
// IMPORTANT gotcha (found live, see PO note on this correction): the
// override tensor matches the layer's FULL output shape [n_embd, n_tokens],
// not just [n_embd] — even though we only care about perturbing the LAST
// token. Filling only the first n_embd floats leaves every OTHER token's
// slice as uninitialized memory, corrupting the whole batch's downstream
// computation (attention/recurrent state). The fix: always supply
// n_tokens*n_embd floats, using each token's own NATURAL captured value for
// every position except the one actually being perturbed.
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

    const int32_t probe_layer = n_layer / 2;
    fprintf(stderr, "Probing intermediate layer %d\n", probe_layer);

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

    // ── Pass 1: natural decode, capture ALL positions at probe_layer ──
    llama_set_layer_output_capture(ctx, &probe_layer, 1);
    llama_batch batch1 = make_batch();
    if (llama_decode(ctx, batch1) != 0) {
        fprintf(stderr, "FAIL: natural decode failed\n");
        return 1;
    }
    // Intermediate-layer capture returns ALL token positions (unlike the
    // last layer, which only captures the single selected output position)
    // — read the full [n_tokens, n_embd] block.
    std::vector<float> h_natural_full((size_t) n_tokens * n_embd);
    for (int i = 0; i < n_tokens; i++) {
        const float * row = llama_get_layer_output(ctx, probe_layer, i);
        if (!row) {
            fprintf(stderr, "FAIL: layer capture returned NULL at position %d\n", i);
            return 1;
        }
        memcpy(h_natural_full.data() + (size_t) i * n_embd, row, n_embd * sizeof(float));
    }
    const float * logits_natural_ptr = llama_get_logits_ith(ctx, n_tokens - 1);
    if (!logits_natural_ptr) {
        fprintf(stderr, "FAIL: natural logits are NULL\n");
        return 1;
    }
    std::vector<float> logits_natural(logits_natural_ptr, logits_natural_ptr + n_vocab);
    llama_set_layer_output_capture(ctx, nullptr, 0);
    llama_batch_free(batch1);

    const float * h_last = h_natural_full.data() + (size_t)(n_tokens - 1) * n_embd;
    double h_norm = 0.0;
    for (int i = 0; i < n_embd; i++) h_norm += (double) h_last[i] * h_last[i];
    h_norm = sqrt(h_norm);
    fprintf(stderr, "Captured h (last token) at layer %d: ||h||=%.4f\n", probe_layer, h_norm);

    // ── Pass 2: round-trip — override with the EXACT captured values (all
    // positions, unmodified) ──
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_set_layer_output_override(ctx, probe_layer, h_natural_full.data(), (int32_t) h_natural_full.size());
    llama_batch batch2 = make_batch();
    if (llama_decode(ctx, batch2) != 0) {
        fprintf(stderr, "FAIL: round-trip decode failed\n");
        return 1;
    }
    const float * logits_roundtrip_ptr = llama_get_logits_ith(ctx, n_tokens - 1);
    if (!logits_roundtrip_ptr) {
        fprintf(stderr, "FAIL: round-trip logits are NULL\n");
        return 1;
    }
    float cos_roundtrip = cosine_similarity(logits_natural.data(), logits_roundtrip_ptr, n_vocab);
    fprintf(stderr, "Round-trip cosine(natural, override-with-same-values) = %.6f\n", cos_roundtrip);
    llama_batch_free(batch2);

    // ── Pass 3: perturbed override — modify ONLY the last token's slice ──
    std::vector<float> h_perturbed_full = h_natural_full;
    float * h_pert_last = h_perturbed_full.data() + (size_t)(n_tokens - 1) * n_embd;
    for (int i = 0; i < n_embd; i++) {
        float dir = ((i * 2654435761u) % 1000) / 500.0f - 1.0f;
        h_pert_last[i] += dir * (float)(0.1 * h_norm) / sqrtf((float) n_embd);
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_set_layer_output_override(ctx, probe_layer, h_perturbed_full.data(), (int32_t) h_perturbed_full.size());
    llama_batch batch3 = make_batch();
    if (llama_decode(ctx, batch3) != 0) {
        fprintf(stderr, "FAIL: perturbed decode failed\n");
        return 1;
    }
    const float * logits_perturbed_ptr = llama_get_logits_ith(ctx, n_tokens - 1);
    if (!logits_perturbed_ptr) {
        fprintf(stderr, "FAIL: perturbed logits are NULL\n");
        return 1;
    }
    float cos_perturbed = cosine_similarity(logits_natural.data(), logits_perturbed_ptr, n_vocab);
    fprintf(stderr, "Perturbed cosine(natural, override-with-perturbed-last-token) = %.6f\n", cos_perturbed);
    llama_batch_free(batch3);

    // ── Verdict ──
    bool ok = true;
    if (cos_roundtrip < 0.999f) {
        fprintf(stderr, "FAIL: round-trip cosine too low (%.6f), expected > 0.999 (override with exact captured values should reproduce the natural result)\n", cos_roundtrip);
        ok = false;
    }
    if (cos_perturbed > 0.9999f) {
        fprintf(stderr, "FAIL: perturbed cosine suspiciously high (%.6f) — override may be silently ignored\n", cos_perturbed);
        ok = false;
    }

    fprintf(stderr, "\n%s\n", ok ? "=== ALL CHECKS PASSED ===" : "=== FAILED ===");

    llama_set_layer_output_override(ctx, -1, nullptr, 0);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
