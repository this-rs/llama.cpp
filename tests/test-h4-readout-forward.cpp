// Test: H4 readout forward pass [obrain]
// Validates llama_model_readout_forward(model, h, logits_out) against REAL
// logits produced by a normal decode on the same captured activation h —
// this is the ground-truth check that the new function actually computes
// what a real inference would produce, before it's trusted as the oracle
// for measuring KL divergence under perturbations to h (PO plan 0648bb5e).
#include "llama.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
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

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 64;
    cparams.n_batch     = 64;
    cparams.n_threads   = 4;
    // logits (not embeddings) — we want the real readout output.
    cparams.embeddings  = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    const int32_t last_layer = n_layer - 1;
    llama_set_layer_output_capture(ctx, &last_layer, 1);

    const char * text = "The capital of France is";
    const int max_tokens = 64;
    std::vector<llama_token> tokens(max_tokens);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_tokens = llama_tokenize(vocab, text, strlen(text), tokens.data(), max_tokens, true, false);
    if (n_tokens < 0) {
        fprintf(stderr, "Tokenization failed\n");
        return 1;
    }
    tokens.resize(n_tokens);
    fprintf(stderr, "Tokenized %d tokens\n", n_tokens);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_tokens - 1); // only need the last token's logits
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    // Real logits from the actual inference. llama_get_logits_ith expects the
    // INPUT batch position (mapped internally via output_ids), not an output
    // slot index — since only position n_tokens-1 has batch.logits[i]=true,
    // that's the index to use here (discovered live: an initial attempt with
    // index 0 failed with "invalid logits id 0, reason: batch.logits[0] != true").
    const float * real_logits = llama_get_logits_ith(ctx, n_tokens - 1);
    if (!real_logits) {
        fprintf(stderr, "FAIL: real logits are NULL\n");
        return 1;
    }

    // Real captured activation h — for the LAST layer, only the last
    // token's output is captured by default (n_outputs=1), so index 0 is
    // the correct (and only valid) index here.
    const float * h = llama_get_layer_output(ctx, last_layer, 0);
    if (!h) {
        fprintf(stderr, "FAIL: captured layer output is NULL\n");
        return 1;
    }

    // Our readout_forward on that same h.
    std::vector<float> my_logits(n_vocab);
    const auto t0 = std::chrono::steady_clock::now();
    int32_t rc = llama_model_readout_forward(model, h, my_logits.data());
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    if (rc != n_vocab) {
        fprintf(stderr, "FAIL: llama_model_readout_forward returned %d, expected %d\n", rc, n_vocab);
        return 1;
    }
    fprintf(stderr, "readout_forward computed in %.3fs\n", secs);

    float cos = cosine_similarity(real_logits, my_logits.data(), n_vocab);
    fprintf(stderr, "Cosine(real_logits, readout_forward_logits) = %.6f\n", cos);

    // Argmax comparison — should pick the exact same next token.
    int argmax_real = 0, argmax_mine = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (real_logits[i] > real_logits[argmax_real]) argmax_real = i;
        if (my_logits[i]   > my_logits[argmax_mine])   argmax_mine = i;
    }
    fprintf(stderr, "Argmax real=%d, argmax mine=%d\n", argmax_real, argmax_mine);

    bool ok = true;
    if (cos < 0.999f) {
        fprintf(stderr, "FAIL: cosine too low (%.6f), expected > 0.999\n", cos);
        ok = false;
    }
    if (argmax_real != argmax_mine) {
        fprintf(stderr, "FAIL: argmax token mismatch\n");
        ok = false;
    }

    fprintf(stderr, "\n%s\n", ok ? "=== ALL CHECKS PASSED ===" : "=== FAILED ===");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
