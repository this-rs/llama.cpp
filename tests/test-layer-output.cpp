// Test: layer output capture API [obrain]
// Validates that intermediate hidden states can be captured during llama_decode.
// Compares the last captured layer output with embeddings from llama_get_embeddings.

#include "llama.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

static float cosine_similarity(const float * a, const float * b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];

    // Initialize
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99; // GPU for test

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    const int n_layer = llama_model_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);
    fprintf(stderr, "Model: n_layer=%d, n_embd=%d\n", n_layer, n_embd);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 64;
    cparams.n_batch   = 64;
    cparams.embeddings = true;  // enable embeddings for comparison
    cparams.n_threads  = 4;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    // Configure layer capture: first, middle, and last layer
    std::vector<int32_t> capture_layers = {0, n_layer / 2, n_layer - 1};
    fprintf(stderr, "Capturing layers: ");
    for (auto l : capture_layers) fprintf(stderr, "%d ", l);
    fprintf(stderr, "\n");

    llama_set_layer_output_capture(ctx, capture_layers.data(), (int32_t)capture_layers.size());

    // Tokenize a short prompt
    const char * text = "Hello world";
    const int max_tokens = 64;
    std::vector<llama_token> tokens(max_tokens);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_tokens = llama_tokenize(vocab, text, strlen(text), tokens.data(), max_tokens, true, false);
    if (n_tokens < 0) {
        fprintf(stderr, "Tokenization failed (need %d tokens)\n", -n_tokens);
        return 1;
    }
    tokens.resize(n_tokens);
    fprintf(stderr, "Tokenized: %d tokens\n", n_tokens);

    // Create batch with all tokens as output
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = true; // mark all as output for embeddings
    }
    batch.n_tokens = n_tokens;

    // Decode
    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        fprintf(stderr, "llama_decode failed: %d\n", ret);
        return 1;
    }

    // Verify captured layers
    fprintf(stderr, "\n=== Layer Output Capture Test ===\n");

    int n_captured = llama_layer_output_n_layers(ctx);
    fprintf(stderr, "Layers captured: %d\n", n_captured);
    if (n_captured != (int)capture_layers.size()) {
        fprintf(stderr, "FAIL: expected %d captured layers, got %d\n", (int)capture_layers.size(), n_captured);
        return 1;
    }

    bool all_ok = true;

    for (auto layer_idx : capture_layers) {
        // Check token 0
        const float * layer_out = llama_get_layer_output(ctx, layer_idx, 0);
        if (!layer_out) {
            fprintf(stderr, "FAIL: layer %d output is NULL\n", layer_idx);
            all_ok = false;
            continue;
        }

        // Compute L2 norm to verify non-zero
        double norm = 0.0;
        for (int j = 0; j < n_embd; j++) {
            norm += (double)layer_out[j] * (double)layer_out[j];
        }
        norm = sqrt(norm);

        fprintf(stderr, "  Layer %2d: L2 norm = %.4f", layer_idx, norm);

        if (norm < 1e-6) {
            fprintf(stderr, " FAIL (zero vector)\n");
            all_ok = false;
        } else {
            fprintf(stderr, " OK\n");
        }
    }

    // Compare last layer output (pre-norm) with embeddings
    // Note: t_embd = output_norm(l_out[last_layer]), so they won't be identical
    //       but should be highly correlated
    const float * embd = llama_get_embeddings_ith(ctx, 0);
    const float * last_layer_out = llama_get_layer_output(ctx, n_layer - 1, 0);
    if (embd && last_layer_out) {
        float cos = cosine_similarity(embd, last_layer_out, n_embd);
        fprintf(stderr, "\n  Cosine(last_layer_out[0], embeddings[0]) = %.6f\n", cos);
        // The embedding = RMS_norm(last_layer_out) * output_norm_weights
        // Since per-element weighting changes direction, cosine > 0.70 is expected
        if (cos > 0.70f) {
            fprintf(stderr, "  PASS: last layer ~ embeddings (cosine=%.4f > 0.70)\n", cos);
        } else {
            fprintf(stderr, "  FAIL: cosine = %.4f (too low, expected > 0.70)\n", cos);
            all_ok = false;
        }
    }

    // Cross-layer comparison: layer 0 should differ from last layer
    const float * first_layer_out = llama_get_layer_output(ctx, 0, 0);
    if (first_layer_out && last_layer_out) {
        float cos = cosine_similarity(first_layer_out, last_layer_out, n_embd);
        fprintf(stderr, "  Cosine(layer_0, layer_%d) = %.6f", n_layer - 1, cos);
        if (cos < 0.98f) {
            fprintf(stderr, " OK (layers differ as expected)\n");
        } else {
            fprintf(stderr, " WARN (layers suspiciously similar)\n");
        }
    }

    fprintf(stderr, "\n%s\n", all_ok ? "=== ALL TESTS PASSED ===" : "=== SOME TESTS FAILED ===");

    // Cleanup
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return all_ok ? 0 : 1;
}
