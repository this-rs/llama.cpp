// Test: verify that custom attention mask and state bias change logits
// Tests both flash_attn=disabled (softmax path) and flash_attn=enabled (auto-fallback)
// Compares logits at the last position — they MUST differ from baseline

#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

struct test_result {
    std::vector<float> logits;
    const char * label;
};

static void print_top5(const llama_vocab * vocab, const char * label, const std::vector<float> & logits) {
    const int n_vocab = (int)logits.size();
    std::vector<std::pair<float, int>> scored(n_vocab);
    for (int v = 0; v < n_vocab; ++v) scored[v] = {logits[v], v};
    std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                     [](auto & a, auto & b) { return a.first > b.first; });
    printf("  %s top-5:", label);
    char buf[128];
    for (int k = 0; k < 5; ++k) {
        int len = llama_token_to_piece(vocab, scored[k].second, buf, sizeof(buf) - 1, 0, false);
        if (len < 0) len = 0;
        buf[len] = '\0';
        printf(" [%d]'%s'=%.2f", scored[k].second, buf, scored[k].first);
    }
    printf("\n");
}

static double l1_distance(const std::vector<float> & a, const std::vector<float> & b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) d += fabs(a[i] - b[i]);
    return d;
}

// Run one full test suite (A=baseline, B=mask, C=bias) with given flash_attn setting
// Returns number of failures
static int run_suite(llama_model * model, const std::vector<llama_token> & tokens,
                     bool flash_attn, const char * suite_name) {
    const int N = (int)tokens.size();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_head  = llama_model_n_head(model);

    printf("=== Suite: %s (flash_attn=%s, N=%d, n_head=%d) ===\n",
           suite_name, flash_attn ? "ON" : "OFF", N, n_head);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = N * 2;
    cparams.n_batch = N;
    cparams.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf("  SKIP: Failed to create context (flash_attn=%s may not be supported)\n\n", flash_attn ? "ON" : "OFF");
        return 0; // not a failure, just unsupported
    }

    auto decode_once = [&](const char * label) -> std::vector<float> {
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(N, 0, 1);
        batch.n_tokens = N;
        for (int i = 0; i < N; ++i) {
            batch.token[i]    = tokens[i];
            batch.pos[i]      = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = (i == N - 1) ? 1 : 0;
        }

        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (rc != 0) {
            printf("  Decode %s FAILED: rc=%d\n", label, rc);
            return {};
        }

        const float * raw = llama_get_logits_ith(ctx, N - 1);
        return std::vector<float>(raw, raw + n_vocab);
    };

    // --- A: baseline (no custom mask, no bias) ---
    llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);
    llama_set_state_bias(ctx, nullptr, 0, 0);
    auto logits_a = decode_once("A-baseline");
    if (logits_a.empty()) { llama_free(ctx); return 1; }

    // --- B: window=2 mask ---
    std::vector<llama_pos> positions(N);
    for (int i = 0; i < N; ++i) positions[i] = i;

    std::vector<float> mask(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            mask[i * N + j] = (j <= i && (i - j) <= 2) ? 0.0f : -INFINITY;
        }
    }

    llama_set_attn_mask(ctx, mask.data(), positions.data(), N, 0, -1);
    llama_set_state_bias(ctx, nullptr, 0, 0);
    auto logits_b = decode_once("B-mask-win2");
    if (logits_b.empty()) { llama_free(ctx); return 1; }

    // --- C: bias +10.0 on position 0 ---
    llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);
    std::vector<float> bias(n_head * N, 0.0f);
    for (int h = 0; h < n_head; ++h) {
        bias[h * N + 0] = 10.0f;
    }
    llama_set_state_bias(ctx, bias.data(), n_head, N);
    auto logits_c = decode_once("C-bias-10");
    if (logits_c.empty()) { llama_free(ctx); return 1; }

    // --- D: mask + bias combined ---
    llama_set_attn_mask(ctx, mask.data(), positions.data(), N, 0, -1);
    llama_set_state_bias(ctx, bias.data(), n_head, N);
    auto logits_d = decode_once("D-mask+bias");
    if (logits_d.empty()) { llama_free(ctx); return 1; }

    // --- Results ---
    double d_ab = l1_distance(logits_a, logits_b);
    double d_ac = l1_distance(logits_a, logits_c);
    double d_ad = l1_distance(logits_a, logits_d);
    double d_bd = l1_distance(logits_b, logits_d); // mask-only vs mask+bias

    printf("  L1 A vs B (mask):       %12.2f\n", d_ab);
    printf("  L1 A vs C (bias):       %12.2f\n", d_ac);
    printf("  L1 A vs D (mask+bias):  %12.2f\n", d_ad);
    printf("  L1 B vs D (bias delta): %12.2f\n", d_bd);

    print_top5(vocab, "A (baseline) ", logits_a);
    print_top5(vocab, "B (window=2) ", logits_b);
    print_top5(vocab, "C (bias=10)  ", logits_c);
    print_top5(vocab, "D (mask+bias)", logits_d);

    int failures = 0;
    auto check = [&](const char * name, double dist) {
        if (dist < 0.001) {
            printf("  FAIL: %s did NOT change logits (L1=%.6f)\n", name, dist);
            failures++;
        } else {
            printf("  PASS: %s (L1=%.2f)\n", name, dist);
        }
    };

    check("Mask effect (A vs B)", d_ab);
    check("Bias effect (A vs C)", d_ac);
    check("Combined effect (A vs D)", d_ad);
    check("Bias adds to mask (B vs D)", d_bd);

    printf("\n");

    llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);
    llama_set_state_bias(ctx, nullptr, 0, 0);
    llama_free(ctx);

    return failures;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [model2.gguf ...]\n", argv[0]);
        return 1;
    }

    const int N = 16;
    int total_failures = 0;

    for (int m = 1; m < argc; ++m) {
        printf("\n########## Model: %s ##########\n\n", argv[m]);

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 99;
        llama_model * model = llama_model_load_from_file(argv[m], mparams);
        if (!model) {
            fprintf(stderr, "Failed to load model: %s\n", argv[m]);
            total_failures++;
            continue;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const char * prompt = "The capital of France is Paris. The largest city in Germany is Berlin. The quick brown fox jumps over the lazy dog near the river bank.";
        std::vector<llama_token> tokens(256);
        int n_tok = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false);
        if (n_tok < N) {
            fprintf(stderr, "Need at least %d tokens, got %d\n", N, n_tok);
            llama_model_free(model);
            total_failures++;
            continue;
        }
        tokens.resize(N);

        // Suite 1: flash_attn OFF (softmax path — gold standard)
        total_failures += run_suite(model, tokens, false, "softmax-path");

        // Suite 2: flash_attn ON (should auto-fallback to softmax when mask/bias is set)
        total_failures += run_suite(model, tokens, true, "flash-attn-fallback");

        llama_model_free(model);
    }

    printf("========================================\n");
    if (total_failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("FAILURES: %d\n", total_failures);
    }

    return total_failures > 0 ? 1 : 0;
}
