// PPL comparison: baseline vs custom mask vs custom mask + state bias
// Tests both trivial (zeros) and non-trivial (window, bias) configurations
//
// Usage: test-ppl-mask <model.gguf> <text_file> [n_ctx=512] [stride=64]

#include "llama.h"
#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <functional>

// Read entire file into string
static std::string read_file(const char * path) {
    std::ifstream f(path);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

// Build positions [0, 1, 2, ..., n-1]
static std::vector<llama_pos> make_positions(int n) {
    std::vector<llama_pos> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = i;
    return pos;
}

// --- Mask generators ---

// All zeros = no modification to default causal mask
static std::vector<float> mask_zeros(int n, int /*n_head*/) {
    return std::vector<float>(n * n, 0.0f);
}

// Window attention: tokens can only attend to the last `window` positions
// mask[i][j] = 0 if j >= i - window, else -inf
static std::vector<float> mask_window(int n, int /*n_head*/, int window) {
    std::vector<float> mask(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // causal: j <= i, window: i - j <= window
            if (j <= i && (i - j) <= window) {
                mask[i * n + j] = 0.0f;
            } else {
                mask[i * n + j] = -INFINITY;
            }
        }
    }
    return mask;
}

// "System prompt boost" mask: first `n_sys` positions always visible (0.0),
// rest follows causal + window. Simulates Obrain's topological mask where
// system/knowledge nodes are always attended to.
static std::vector<float> mask_sys_window(int n, int /*n_head*/, int n_sys, int window) {
    std::vector<float> mask(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            bool causal = (j <= i);
            bool in_sys = (j < n_sys);
            bool in_window = ((i - j) <= window);
            if (causal && (in_sys || in_window)) {
                mask[i * n + j] = 0.0f;
            } else {
                mask[i * n + j] = -INFINITY;
            }
        }
    }
    return mask;
}

// --- Bias generators ---

// Uniform positive bias on first n_boost positions (simulates saliency boost)
static std::vector<float> bias_boost_prefix(int n_head, int n_kv, int n_boost, float magnitude) {
    std::vector<float> bias(n_head * n_kv, 0.0f);
    for (int h = 0; h < n_head; ++h) {
        for (int j = 0; j < n_boost && j < n_kv; ++j) {
            bias[h * n_kv + j] = magnitude;
        }
    }
    return bias;
}

// Decaying bias: position j gets magnitude * exp(-j / tau)
static std::vector<float> bias_decay(int n_head, int n_kv, float magnitude, float tau) {
    std::vector<float> bias(n_head * n_kv, 0.0f);
    for (int h = 0; h < n_head; ++h) {
        for (int j = 0; j < n_kv; ++j) {
            bias[h * n_kv + j] = magnitude * expf(-(float)j / tau);
        }
    }
    return bias;
}

// --- Test configuration ---

struct test_config {
    const char * name;
    // nullptr = no custom mask
    std::vector<float> mask;
    bool has_mask;
    int n_head_groups; // 0 = broadcast

    // nullptr = no bias
    std::vector<float> bias;
    bool has_bias;
    int bias_n_head;
};

struct ppl_result {
    double ppl;
    int n_chunks;
};

// Compute perplexity for a given configuration
static ppl_result compute_ppl(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::vector<llama_token> & tokens,
    int n_ctx,
    int stride,
    const test_config & cfg
) {
    const int n_tokens = (int)tokens.size();

    double nll_sum = 0.0;
    int nll_count = 0;
    int n_chunks = 0;

    auto positions = make_positions(n_ctx);

    for (int start = 0; start + n_ctx <= n_tokens; start += stride) {
        llama_memory_clear(llama_get_memory(ctx), true);

        // Set mask
        if (cfg.has_mask) {
            llama_set_attn_mask(ctx, cfg.mask.data(), positions.data(), n_ctx, cfg.n_head_groups, -1);
        } else {
            llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);
        }

        // Set bias
        if (cfg.has_bias) {
            llama_set_state_bias(ctx, cfg.bias.data(), cfg.bias_n_head, n_ctx);
        } else {
            llama_set_state_bias(ctx, nullptr, 0, 0);
        }

        // Build batch
        llama_batch batch = llama_batch_init(n_ctx, 0, 1);
        batch.n_tokens = n_ctx;
        for (int i = 0; i < n_ctx; ++i) {
            batch.token[i]     = tokens[start + i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i >= n_ctx / 2) ? 1 : 0;
        }

        int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "  [%s] llama_decode failed at start=%d (rc=%d)\n", cfg.name, start, rc);
            llama_batch_free(batch);
            continue;
        }

        // NLL: logits[i] predicts token[i+1]
        int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = n_ctx / 2; i < n_ctx - 1; ++i) {
            const float * logits = llama_get_logits_ith(ctx, i);
            if (!logits) continue;

            llama_token target = tokens[start + i + 1];

            float max_logit = -INFINITY;
            for (int v = 0; v < n_vocab; ++v) {
                if (logits[v] > max_logit) max_logit = logits[v];
            }

            double sum_exp = 0.0;
            for (int v = 0; v < n_vocab; ++v) {
                sum_exp += exp((double)(logits[v] - max_logit));
            }

            double log_prob = (double)(logits[target] - max_logit) - log(sum_exp);
            nll_sum -= log_prob;
            nll_count++;
        }

        n_chunks++;
        llama_batch_free(batch);
    }

    ppl_result res;
    res.ppl = (nll_count > 0) ? exp(nll_sum / nll_count) : 0.0;
    res.n_chunks = n_chunks;
    return res;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <text_file> [n_ctx=512] [stride=64]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * text_path  = argv[2];
    const int n_ctx   = argc > 3 ? atoi(argv[3]) : 512;
    const int stride  = argc > 4 ? atoi(argv[4]) : 64;

    std::string text = read_file(text_path);
    if (text.empty()) {
        fprintf(stderr, "Failed to read text file: %s\n", text_path);
        return 1;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = n_ctx * 2;
    cparams.n_batch = n_ctx;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); llama_model_free(model); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_head = llama_model_n_head(model);

    std::vector<llama_token> tokens(text.size() + 100);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) { fprintf(stderr, "Tokenization failed\n"); llama_free(ctx); llama_model_free(model); return 1; }
    tokens.resize(n_tokens);

    printf("Model:    %s\n", model_path);
    printf("Text:     %s (%d tokens)\n", text_path, n_tokens);
    printf("n_ctx:    %d, stride: %d, n_head: %d\n", n_ctx, stride, n_head);
    printf("flash:    OFF (consistent for all modes)\n\n");

    // =====================================================================
    // Define test configurations
    // =====================================================================
    std::vector<test_config> configs;

    // A) Baseline — no custom mask, no bias
    {
        test_config c;
        c.name = "A) Baseline";
        c.has_mask = false; c.n_head_groups = 0;
        c.has_bias = false; c.bias_n_head = 0;
        configs.push_back(std::move(c));
    }

    // B) Causal zeros mask — should be bit-exact same as A
    {
        test_config c;
        c.name = "B) Mask: causal zeros";
        c.mask = mask_zeros(n_ctx, n_head);
        c.has_mask = true; c.n_head_groups = 0;
        c.has_bias = false; c.bias_n_head = 0;
        configs.push_back(std::move(c));
    }

    // C) Mask zeros + zero bias — should be bit-exact same as A
    {
        test_config c;
        c.name = "C) Mask zeros + zero bias";
        c.mask = mask_zeros(n_ctx, n_head);
        c.has_mask = true; c.n_head_groups = 0;
        c.bias.resize(n_head * n_ctx, 0.0f);
        c.has_bias = true; c.bias_n_head = n_head;
        configs.push_back(std::move(c));
    }

    // D) Window=256 — moderate restriction (half context)
    {
        test_config c;
        c.name = "D) Mask: window=256";
        c.mask = mask_window(n_ctx, n_head, 256);
        c.has_mask = true; c.n_head_groups = 0;
        c.has_bias = false; c.bias_n_head = 0;
        configs.push_back(std::move(c));
    }

    // E) Window=64 — aggressive restriction
    {
        test_config c;
        c.name = "E) Mask: window=64";
        c.mask = mask_window(n_ctx, n_head, 64);
        c.has_mask = true; c.n_head_groups = 0;
        c.has_bias = false; c.bias_n_head = 0;
        configs.push_back(std::move(c));
    }

    // F) System prompt window: first 32 tokens always visible + window=128
    //    Simulates Obrain's topological mask (knowledge nodes = always visible)
    {
        test_config c;
        c.name = "F) Mask: sys=32 + win=128";
        c.mask = mask_sys_window(n_ctx, n_head, 32, 128);
        c.has_mask = true; c.n_head_groups = 0;
        c.has_bias = false; c.bias_n_head = 0;
        configs.push_back(std::move(c));
    }

    // G) Prefix bias: boost first 32 positions by +0.5 (mild saliency)
    {
        test_config c;
        c.name = "G) Bias: prefix=32, mag=0.5";
        c.has_mask = false; c.n_head_groups = 0;
        c.bias = bias_boost_prefix(n_head, n_ctx, 32, 0.5f);
        c.has_bias = true; c.bias_n_head = n_head;
        configs.push_back(std::move(c));
    }

    // H) Prefix bias: boost first 32 positions by +2.0 (strong saliency)
    {
        test_config c;
        c.name = "H) Bias: prefix=32, mag=2.0";
        c.has_mask = false; c.n_head_groups = 0;
        c.bias = bias_boost_prefix(n_head, n_ctx, 32, 2.0f);
        c.has_bias = true; c.bias_n_head = n_head;
        configs.push_back(std::move(c));
    }

    // I) Decay bias: magnitude=1.0, tau=64 (recent tokens boosted)
    {
        test_config c;
        c.name = "I) Bias: decay mag=1.0 tau=64";
        c.has_mask = false; c.n_head_groups = 0;
        c.bias = bias_decay(n_head, n_ctx, 1.0f, 64.0f);
        c.has_bias = true; c.bias_n_head = n_head;
        configs.push_back(std::move(c));
    }

    // J) Combo: sys=32 + win=128 mask + prefix bias=0.5
    //    Full Obrain-style: topological mask + saliency bias
    {
        test_config c;
        c.name = "J) Combo: sys+win mask + bias";
        c.mask = mask_sys_window(n_ctx, n_head, 32, 128);
        c.has_mask = true; c.n_head_groups = 0;
        c.bias = bias_boost_prefix(n_head, n_ctx, 32, 0.5f);
        c.has_bias = true; c.bias_n_head = n_head;
        configs.push_back(std::move(c));
    }

    // =====================================================================
    // Run all tests
    // =====================================================================
    std::vector<ppl_result> results;

    for (size_t i = 0; i < configs.size(); ++i) {
        printf("=== %s ===\n", configs[i].name);
        fflush(stdout);
        auto res = compute_ppl(ctx, vocab, tokens, n_ctx, stride, configs[i]);
        printf("  PPL = %.4f  (%d chunks)\n\n", res.ppl, res.n_chunks);
        fflush(stdout);
        results.push_back(res);
    }

    // =====================================================================
    // Summary table
    // =====================================================================
    double baseline = results[0].ppl;

    printf("╔══════════════════════════════════════╦══════════╦══════════╦══════════╗\n");
    printf("║ %-36s ║ %8s ║ %8s ║ %8s ║\n", "Mode", "PPL", "Delta", "Delta%");
    printf("╠══════════════════════════════════════╬══════════╬══════════╬══════════╣\n");

    for (size_t i = 0; i < configs.size(); ++i) {
        double delta = results[i].ppl - baseline;
        double delta_pct = (baseline > 0) ? delta / baseline * 100.0 : 0.0;
        if (i == 0) {
            printf("║ %-36s ║ %8.4f ║ %8s ║ %8s ║\n", configs[i].name, results[i].ppl, "-", "-");
        } else {
            printf("║ %-36s ║ %8.4f ║ %+8.4f ║ %+7.2f%% ║\n",
                   configs[i].name, results[i].ppl, delta, delta_pct);
        }
    }

    printf("╚══════════════════════════════════════╩══════════╩══════════╩══════════╝\n\n");

    // Validation
    bool pass = true;
    // B and C must be bit-exact with A
    if (fabs(results[1].ppl - baseline) > 0.001) { printf("FAIL: B diverges from baseline\n"); pass = false; }
    if (fabs(results[2].ppl - baseline) > 0.001) { printf("FAIL: C diverges from baseline\n"); pass = false; }
    // D-J should show measurable but bounded effects
    for (size_t i = 3; i < results.size(); ++i) {
        double delta_pct = (results[i].ppl - baseline) / baseline * 100.0;
        if (results[i].ppl <= 0 || delta_pct > 200.0) {
            printf("FAIL: %s has extreme PPL (%.4f, %+.1f%%)\n", configs[i].name, results[i].ppl, delta_pct);
            pass = false;
        }
    }

    if (pass) {
        printf("ALL PASS: Trivial configs bit-exact, non-trivial configs bounded\n");
    }

    llama_set_attn_mask(ctx, nullptr, nullptr, 0, 0, -1);
    llama_set_state_bias(ctx, nullptr, 0, 0);
    llama_free(ctx);
    llama_model_free(model);

    return pass ? 0 : 1;
}
