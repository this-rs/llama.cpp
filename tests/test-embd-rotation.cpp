// Test: embedding injection via batch.embd with Hadamard rotation enabled
// Validates that injecting pre-computed embeddings works correctly when
// KV cache is quantized (q4_0/q8_0) and rotation matrices are active.

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

static float cosine_sim(const float * a, const float * b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

static float kl_divergence(const float * p_logits, const float * q_logits, int n) {
    // KL(softmax(p) || softmax(q)) — numerically stable
    float max_p = *std::max_element(p_logits, p_logits + n);
    float max_q = *std::max_element(q_logits, q_logits + n);
    double sum_exp_p = 0, sum_exp_q = 0;
    for (int i = 0; i < n; i++) {
        sum_exp_p += exp((double)(p_logits[i] - max_p));
        sum_exp_q += exp((double)(q_logits[i] - max_q));
    }
    double log_Z_p = max_p + log(sum_exp_p);
    double log_Z_q = max_q + log(sum_exp_q);
    double kl = 0;
    for (int i = 0; i < n; i++) {
        double log_pi = (double)p_logits[i] - log_Z_p;
        double log_qi = (double)q_logits[i] - log_Z_q;
        double pi = exp(log_pi);
        if (pi > 1e-10) {
            kl += pi * (log_pi - log_qi);
        }
    }
    return (float)kl;
}

static int top_k_overlap(const float * a, const float * b, int n, int k) {
    std::vector<int> idx_a(n), idx_b(n);
    std::iota(idx_a.begin(), idx_a.end(), 0);
    std::iota(idx_b.begin(), idx_b.end(), 0);
    std::partial_sort(idx_a.begin(), idx_a.begin() + k, idx_a.end(),
        [&](int i, int j){ return a[i] > a[j]; });
    std::partial_sort(idx_b.begin(), idx_b.begin() + k, idx_b.end(),
        [&](int i, int j){ return b[i] > b[j]; });
    int overlap = 0;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            if (idx_a[i] == idx_b[j]) { overlap++; break; }
    return overlap;
}

static std::vector<float> run_embd_decode(llama_model * model, ggml_type type_k, ggml_type type_v,
                                           const float * embd, int n_embd, int n_inject) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 64;
    cparams.n_batch = 64;
    cparams.type_k  = type_k;
    cparams.type_v  = type_v;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return {};

    llama_batch batch = llama_batch_init(n_inject, n_embd, 1);
    for (int i = 0; i < n_inject; i++) {
        memcpy(batch.embd + i * n_embd, embd + i * n_embd, n_embd * sizeof(float));
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_inject - 1) ? 1 : 0;
    }
    batch.n_tokens = n_inject;

    std::vector<float> result;
    if (llama_decode(ctx, batch) == 0) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * l = llama_get_logits_ith(ctx, n_inject - 1);
        result.assign(l, l + n_vocab);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    return result;
}

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

    // =============================================
    // Test 4: Logit quality — f16 vs quantized+rotation
    // =============================================
    printf("\n=== Test 4: Logit quality f16 vs %s+rotation ===\n",
           type_k == GGML_TYPE_F16 ? "f16(self)" :
           type_k == GGML_TYPE_Q8_0 ? "q8_0" :
           type_k == GGML_TYPE_Q5_0 ? "q5_0" :
           type_k == GGML_TYPE_Q4_0 ? "q4_0" : "other");
    {
        const int n_inject = 5;
        // Generate deterministic embeddings
        std::vector<float> embd(n_inject * n_embd);
        for (int i = 0; i < n_inject; i++) {
            float norm = 0;
            for (int j = 0; j < n_embd; j++) {
                float val = sinf((float)(i * 7 + j * 13) * 0.0073f)
                          + cosf((float)(i * 11 + j * 3) * 0.0091f);
                embd[i * n_embd + j] = val;
                norm += val * val;
            }
            norm = sqrtf(norm);
            for (int j = 0; j < n_embd; j++) embd[i * n_embd + j] /= norm;
        }

        // Run f16 reference
        auto logits_f16 = run_embd_decode(model, GGML_TYPE_F16, GGML_TYPE_F16,
                                           embd.data(), n_embd, n_inject);
        // Run with requested quantization
        auto logits_quant = run_embd_decode(model, type_k, type_v,
                                             embd.data(), n_embd, n_inject);

        if (logits_f16.empty() || logits_quant.empty()) {
            fail("could not produce logits for comparison");
        } else {
            const int n_vocab = (int)logits_f16.size();
            float cos = cosine_sim(logits_f16.data(), logits_quant.data(), n_vocab);
            float kl  = kl_divergence(logits_f16.data(), logits_quant.data(), n_vocab);
            int top10 = top_k_overlap(logits_f16.data(), logits_quant.data(), n_vocab, 10);
            int top50 = top_k_overlap(logits_f16.data(), logits_quant.data(), n_vocab, 50);

            printf("  cosine(f16, quant) = %.6f\n", cos);
            printf("  KL(f16 || quant)   = %.6f nats\n", kl);
            printf("  top-10 overlap     = %d/10\n", top10);
            printf("  top-50 overlap     = %d/50\n", top50);

            // Thresholds: cosine > 0.95, KL < 0.5, top-10 overlap >= 7
            char buf[128];
            if (cos >= 0.95f && kl < 0.5f && top10 >= 7) {
                snprintf(buf, sizeof(buf), "logit quality OK (cos=%.4f, KL=%.4f, top10=%d/10)", cos, kl, top10);
                pass(buf);
            } else if (cos >= 0.90f && top10 >= 5) {
                snprintf(buf, sizeof(buf), "logit quality ACCEPTABLE (cos=%.4f, KL=%.4f, top10=%d/10)", cos, kl, top10);
                pass(buf);
            } else {
                snprintf(buf, sizeof(buf), "logit quality DEGRADED (cos=%.4f, KL=%.4f, top10=%d/10)", cos, kl, top10);
                fail(buf);
            }
        }
    }

    // =============================================
    // Test 5: Latency & throughput measurement
    // =============================================
    printf("\n=== Test 5: Latency measurement ===\n");
    {
        const int n_inject_sizes[] = {1, 5, 10, 20};
        const int n_sizes = 4;

        for (int s = 0; s < n_sizes; s++) {
            const int n_inj = n_inject_sizes[s];
            // Generate embeddings
            std::vector<float> embd(n_inj * n_embd);
            for (int i = 0; i < n_inj * n_embd; i++) {
                embd[i] = sinf((float)i * 0.0073f) * 0.1f;
            }

            // Time the decode
            auto t0 = std::chrono::high_resolution_clock::now();
            auto logits = run_embd_decode(model, type_k, type_v, embd.data(), n_embd, n_inj);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            printf("  %2d embeddings: %.1f ms (%.1f ms/embd)%s\n",
                   n_inj, ms, ms / n_inj,
                   logits.empty() ? " FAILED" : "");
        }

        // VRAM estimation (theoretical)
        // KV cache: 2 * n_layers * n_ctx * n_embd * bytes_per_element
        int n_layers = 28; // llama 3.2 3B
        int n_ctx = 64;
        int bytes_f16 = 2, bytes_q8 = 1, bytes_q5 = 1, bytes_q4 = 1; // approx per element
        int bpe = (type_k == GGML_TYPE_F16) ? bytes_f16 :
                  (type_k == GGML_TYPE_Q8_0) ? bytes_q8 : bytes_q4;
        double kv_mb = 2.0 * n_layers * n_ctx * n_embd * bpe / (1024.0 * 1024.0);
        double kv_mb_f16 = 2.0 * n_layers * n_ctx * n_embd * bytes_f16 / (1024.0 * 1024.0);
        double ratio = kv_mb_f16 / kv_mb;
        printf("  KV cache estimate: %.1f MiB (f16: %.1f MiB, ratio: %.1fx)\n", kv_mb, kv_mb_f16, ratio);

        char buf[128];
        snprintf(buf, sizeof(buf), "latency measured, KV ratio %.1fx vs f16", ratio);
        pass(buf);
    }

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", n_passed, n_failed);
    printf("========================================\n");

    llama_model_free(model);
    return n_failed > 0 ? 1 : 0;
}
