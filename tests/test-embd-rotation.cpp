// Test: embedding injection via batch.embd with Hadamard rotation enabled
// Validates that injecting pre-computed embeddings works correctly when
// KV cache is quantized (q4_0/q8_0) and rotation matrices are active.

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static ggml_type parse_ggml_type(const char * s) {
    if (strcmp(s, "f16")  == 0) return GGML_TYPE_F16;
    if (strcmp(s, "q8_0") == 0) return GGML_TYPE_Q8_0;
    if (strcmp(s, "q5_0") == 0) return GGML_TYPE_Q5_0;
    if (strcmp(s, "q4_0") == 0) return GGML_TYPE_Q4_0;
    return GGML_TYPE_F16;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [-ctk type] [-ctv type]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], "-ctk") == 0) { type_k = parse_ggml_type(argv[++i]); }
        if (strcmp(argv[i], "-ctv") == 0) { type_v = parse_ggml_type(argv[++i]); }
    }

    printf("KV cache types: K=%d V=%d\n", type_k, type_v);

    // Load model
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const int n_embd = llama_model_n_embd(model);
    printf("n_embd = %d\n", n_embd);

    int n_passed = 0;
    int n_failed = 0;

    auto pass = [&](const char * msg) { printf("PASS: %s\n", msg); n_passed++; };
    auto fail = [&](const char * msg) { printf("FAIL: %s\n", msg); n_failed++; };

    // =============================================
    // Test 1: Token-based decode (baseline)
    // =============================================
    printf("\n=== Test 1: Token-based decode baseline ===\n");
    std::vector<float> logits_token;
    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx   = 64;
        cparams.n_batch = 64;
        cparams.type_k  = type_k;
        cparams.type_v  = type_v;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) { fail("Failed to create context"); return 1; }

        const llama_vocab * vocab = llama_model_get_vocab(model);

        // Tokenize "Hello world"
        std::vector<llama_token> tokens(64);
        int n_tok = llama_tokenize(vocab, "Hello world", strlen("Hello world"), tokens.data(), tokens.size(), true, false);
        if (n_tok < 0) { n_tok = -n_tok; tokens.resize(n_tok); llama_tokenize(vocab, "Hello world", strlen("Hello world"), tokens.data(), tokens.size(), true, false); }
        else { tokens.resize(n_tok); }

        // Decode via token batch
        llama_batch batch = llama_batch_init(n_tok, 0, 1);
        for (int i = 0; i < n_tok; i++) {
            batch.token[i]    = tokens[i];
            batch.pos[i]      = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = (i == n_tok - 1) ? 1 : 0;
        }
        batch.n_tokens = n_tok;

        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            char buf[64]; snprintf(buf, sizeof(buf), "token decode failed: ret=%d", ret);
            fail(buf);
        } else {
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            const float * l = llama_get_logits_ith(ctx, n_tok - 1);
            logits_token.assign(l, l + n_vocab);
            char buf[64]; snprintf(buf, sizeof(buf), "token decode OK (%d tokens, %d logits)", n_tok, n_vocab);
            pass(buf);
        }

        llama_batch_free(batch);
        llama_free(ctx);
    }

    // =============================================
    // Test 2: Embedding injection via batch.embd
    // =============================================
    printf("\n=== Test 2: Embedding injection via batch.embd ===\n");
    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx   = 64;
        cparams.n_batch = 64;
        cparams.type_k  = type_k;
        cparams.type_v  = type_v;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) { fail("Failed to create context"); return 1; }

        // Create a batch with embeddings (not tokens)
        const int n_inject = 3; // inject 3 embedding vectors
        llama_batch batch = llama_batch_init(n_inject, n_embd, 1);

        if (batch.embd == nullptr) {
            fail("batch.embd is null after llama_batch_init with embd > 0");
        } else {
            // Fill with random-ish embeddings (normalized)
            for (int i = 0; i < n_inject; i++) {
                for (int j = 0; j < n_embd; j++) {
                    float val = sinf((float)(i * n_embd + j) * 0.01f);
                    batch.embd[i * n_embd + j] = val * 0.1f;
                }
                batch.pos[i]      = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]   = (i == n_inject - 1) ? 1 : 0;
            }
            batch.n_tokens = n_inject;

            int ret = llama_decode(ctx, batch);
            if (ret != 0) {
                char buf[64]; snprintf(buf, sizeof(buf), "embd decode failed: ret=%d", ret);
                fail(buf);
            } else {
                const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
                const float * l = llama_get_logits_ith(ctx, n_inject - 1);

                // Check logits are valid (not NaN, not all zeros)
                bool has_nan = false;
                bool all_zero = true;
                for (int i = 0; i < n_vocab && i < 100; i++) {
                    if (std::isnan(l[i])) has_nan = true;
                    if (l[i] != 0.0f) all_zero = false;
                }

                if (has_nan) {
                    fail("embd decode produced NaN logits");
                } else if (all_zero) {
                    fail("embd decode produced all-zero logits");
                } else {
                    pass("embd decode OK — valid logits produced");
                }
            }
        }

        llama_batch_free(batch);
        llama_free(ctx);
    }

    // =============================================
    // Test 3: Mixed token + embedding in same context
    // =============================================
    printf("\n=== Test 3: Token decode then embedding injection in same context ===\n");
    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx   = 64;
        cparams.n_batch = 64;
        cparams.type_k  = type_k;
        cparams.type_v  = type_v;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) { fail("Failed to create context"); return 1; }

        const llama_vocab * vocab = llama_model_get_vocab(model);

        // First: decode some tokens
        std::vector<llama_token> tokens(64);
        int n_tok = llama_tokenize(vocab, "Hello", strlen("Hello"), tokens.data(), tokens.size(), true, false);
        if (n_tok < 0) { n_tok = -n_tok; tokens.resize(n_tok); llama_tokenize(vocab, "Hello", strlen("Hello"), tokens.data(), tokens.size(), true, false); }
        else { tokens.resize(n_tok); }

        llama_batch batch_tok = llama_batch_init(n_tok, 0, 1);
        for (int i = 0; i < n_tok; i++) {
            batch_tok.token[i]    = tokens[i];
            batch_tok.pos[i]      = i;
            batch_tok.n_seq_id[i] = 1;
            batch_tok.seq_id[i][0] = 0;
            batch_tok.logits[i]   = 0;
        }
        batch_tok.n_tokens = n_tok;

        int ret1 = llama_decode(ctx, batch_tok);
        llama_batch_free(batch_tok);

        if (ret1 != 0) {
            char buf[64]; snprintf(buf, sizeof(buf), "initial token decode failed: ret=%d", ret1);
            fail(buf);
        } else {
            // Now inject embeddings at positions after the tokens
            const int n_inject = 2;
            llama_batch batch_embd = llama_batch_init(n_inject, n_embd, 1);

            for (int i = 0; i < n_inject; i++) {
                for (int j = 0; j < n_embd; j++) {
                    batch_embd.embd[i * n_embd + j] = cosf((float)(i * n_embd + j) * 0.02f) * 0.1f;
                }
                batch_embd.pos[i]      = n_tok + i; // positions after tokens
                batch_embd.n_seq_id[i] = 1;
                batch_embd.seq_id[i][0] = 0;
                batch_embd.logits[i]   = (i == n_inject - 1) ? 1 : 0;
            }
            batch_embd.n_tokens = n_inject;

            int ret2 = llama_decode(ctx, batch_embd);
            llama_batch_free(batch_embd);

            if (ret2 != 0) {
                char buf[64]; snprintf(buf, sizeof(buf), "embd injection after tokens failed: ret=%d", ret2);
                fail(buf);
            } else {
                const float * l = llama_get_logits_ith(ctx, n_inject - 1);
                bool has_nan = false;
                for (int i = 0; i < 100; i++) {
                    if (std::isnan(l[i])) { has_nan = true; break; }
                }
                if (has_nan) {
                    fail("mixed token+embd produced NaN logits");
                } else {
                    pass("mixed token+embd decode OK — embeddings injected after tokens");
                }
            }
        }

        llama_free(ctx);
    }

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", n_passed, n_failed);
    printf("========================================\n");

    llama_model_free(model);
    return n_failed > 0 ? 1 : 0;
}
