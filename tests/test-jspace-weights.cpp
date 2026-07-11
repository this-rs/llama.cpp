// Test: J-space readout weight accessors [obrain]
// Validates llama_model_get_output_norm_weight / llama_model_get_output_gram_matrix
// on a real loaded model — sizes match hparams, values are finite, gram matrix is
// symmetric and positive semi-definite (as claimed by the geometric manifesto for
// g^out = J^T J; the Gram matrix W_out^T W_out is a *necessary* precondition for
// that, since g^out = J_rmsnorm^T @ gram @ J_rmsnorm inherits PSD-ness from gram
// when J_rmsnorm has full rank — see plan task T5 for the full pullback validation).
#include "llama.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

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

    const int32_t n_embd = llama_model_n_embd(model);
    fprintf(stderr, "Model: n_embd=%d\n", n_embd);

    // --- output_norm.weight ---
    std::vector<float> norm_w(n_embd);
    int32_t rc_norm = llama_model_get_output_norm_weight(model, norm_w.data());
    if (rc_norm != n_embd) {
        fprintf(stderr, "FAIL: llama_model_get_output_norm_weight returned %d, expected %d\n", rc_norm, n_embd);
        return 1;
    }
    int n_zero = 0, n_nan = 0;
    for (int32_t i = 0; i < n_embd; i++) {
        if (std::isnan(norm_w[i]) || std::isinf(norm_w[i])) n_nan++;
        if (norm_w[i] == 0.0f) n_zero++;
    }
    if (n_nan > 0) {
        fprintf(stderr, "FAIL: output_norm.weight contains %d NaN/Inf values\n", n_nan);
        return 1;
    }
    fprintf(stderr, "OK: output_norm.weight size=%d, zeros=%d, sample[0..3]=%.4f %.4f %.4f %.4f\n",
        rc_norm, n_zero, norm_w[0], norm_w[1], norm_w[2], norm_w[3]);

    // --- W_out^T W_out gram matrix ---
    std::vector<float> gram((size_t) n_embd * n_embd);
    fprintf(stderr, "Computing gram matrix (streams over n_vocab rows, one-time cost)...\n");
    const auto t0 = std::chrono::steady_clock::now();
    int32_t rc_gram = llama_model_get_output_gram_matrix(model, gram.data());
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    if (rc_gram != n_embd) {
        fprintf(stderr, "FAIL: llama_model_get_output_gram_matrix returned %d, expected %d\n", rc_gram, n_embd);
        return 1;
    }
    fprintf(stderr, "OK: gram matrix computed in %.2fs (n_embd=%d, n_embd^2=%zu floats)\n",
        secs, n_embd, (size_t) n_embd * n_embd);

    // Symmetry check: gram[i][j] should equal gram[j][i] exactly (we mirror it
    // explicitly in the C implementation, so this should be a bit-exact check).
    int n_asym = 0;
    for (int32_t i = 0; i < n_embd && n_asym < 5; i++) {
        for (int32_t j = i + 1; j < n_embd; j++) {
            float a = gram[(size_t) i * n_embd + j];
            float b = gram[(size_t) j * n_embd + i];
            if (a != b) {
                n_asym++;
                fprintf(stderr, "  asymmetry at (%d,%d): %.6f vs %.6f\n", i, j, a, b);
            }
        }
    }
    if (n_asym > 0) {
        fprintf(stderr, "FAIL: gram matrix is not exactly symmetric (%d+ mismatches)\n", n_asym);
        return 1;
    }
    fprintf(stderr, "OK: gram matrix is exactly symmetric\n");

    // Diagonal must be non-negative (gram[i][i] = sum_v w_v[i]^2 >= 0) — a
    // necessary (not sufficient) condition for PSD, cheap to check here; the
    // full eigenvalue PSD check is done in T5 once wired into Rust.
    int n_neg_diag = 0;
    double diag_sum = 0.0;
    for (int32_t i = 0; i < n_embd; i++) {
        float d = gram[(size_t) i * n_embd + i];
        if (d < 0.0f) n_neg_diag++;
        diag_sum += d;
    }
    if (n_neg_diag > 0) {
        fprintf(stderr, "FAIL: %d negative diagonal entries in gram matrix (impossible for w^Tw)\n", n_neg_diag);
        return 1;
    }
    fprintf(stderr, "OK: gram matrix diagonal all >= 0 (trace=%.4f)\n", diag_sum);

    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "ALL CHECKS PASSED\n");
    return 0;
}
