#!/bin/bash
# Run attention mask tests across multiple models
# Compares knowledge isolation and reasoning capabilities

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build/bin"

# Models to test (name → gguf path)
declare -A MODELS
MODELS=(
    ["llama3.2-3b"]="/Users/triviere/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff"
    ["llama3-8b"]="/Users/triviere/.ollama/models/blobs/sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa"
    ["deepseek-r1-7b"]="/Users/triviere/.ollama/models/blobs/sha256-96c415656d377afbff962f6cdb2394ab092ccbcbaab4b82525bc4ca800fe8a49"
    ["gemma3n-8b"]="/Users/triviere/.ollama/models/blobs/sha256-38e8dcc30df4eb0e29eaf5c74ba6ce3f2cd66badad50768fc14362acfb8b8cb6"
    ["mistral-small-24b"]="/Users/triviere/.ollama/models/blobs/sha256-102a747c137683e81d431dab05d8f2158df4ab6f162f8f9019425a43d51e0e9f"
    ["magistral-24b"]="/Users/triviere/.ollama/models/blobs/sha256-641615e9986bc8687f936cd87c586bdd92d338172c4180963080e48b8e84ec36"
    ["qwen3-32b"]="/Users/triviere/.ollama/models/blobs/sha256-3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312"
    ["deepseek-r1-32b"]="/Users/triviere/.ollama/models/blobs/sha256-6150cb382311b69f09cc0f9a1b69fc029cbd742b66bb8ec531aa5ecf5c613e93"
    ["gemma3-27b"]="/Users/triviere/.ollama/models/blobs/sha256-afa0ea2ef463c87a1eebb9af070e76a353107493b5d9a62e5e66f65a65409541"
)

# Order for display (small → large)
ORDER=(
    "llama3.2-3b"
    "llama3-8b"
    "deepseek-r1-7b"
    "gemma3n-8b"
    "mistral-small-24b"
    "magistral-24b"
    "gemma3-27b"
    "qwen3-32b"
    "deepseek-r1-32b"
)

# Tests to run
TESTS=("test-attn-mask-topology" "test-attn-mask-reasoning")

# Output directory
OUT_DIR="/tmp/attn-mask-results-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Attention Mask Multi-Model Test Suite                   ║"
echo "║  Models: ${#ORDER[@]}   Tests: ${#TESTS[@]}                              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Results directory: $OUT_DIR"
echo ""

# Parse results from test output
parse_results() {
    local output="$1"
    local pass=$(grep -c "PASS:" <<< "$output" 2>/dev/null || echo "0")
    local fail=$(grep -c "FAIL:" <<< "$output" 2>/dev/null || echo "0")
    local info=$(grep -c "INFO:" <<< "$output" 2>/dev/null || echo "0")
    echo "$pass/$fail/$info"
}

# Run all tests
for model_name in "${ORDER[@]}"; do
    model_path="${MODELS[$model_name]}"
    if [ ! -f "$model_path" ]; then
        echo "⚠ SKIP: $model_name — file not found"
        continue
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ $model_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for test in "${TESTS[@]}"; do
        test_bin="$BUILD_DIR/$test"
        if [ ! -x "$test_bin" ]; then
            echo "  ⚠ $test: binary not found, skipping"
            continue
        fi

        echo -n "  $test ... "
        out_file="$OUT_DIR/${model_name}_${test}.log"

        # Run with timeout, capture stdout, discard stderr (model loading noise)
        if timeout 300 "$test_bin" "$model_path" > "$out_file" 2>/dev/null; then
            result=$(parse_results "$(cat "$out_file")")
            echo "✓ pass/fail/info: $result"
        else
            exit_code=$?
            if [ $exit_code -eq 124 ]; then
                echo "⏱ TIMEOUT (300s)"
                echo "TIMEOUT" > "$out_file"
            else
                result=$(parse_results "$(cat "$out_file")")
                echo "✗ pass/fail/info: $result"
            fi
        fi
    done
    echo ""
done

# Summary table
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  SUMMARY TABLE                                                              ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
printf "║ %-22s │ %-20s │ %-20s ║\n" "Model" "Topology (P/F/I)" "Reasoning (P/F/I)"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"

for model_name in "${ORDER[@]}"; do
    topo_file="$OUT_DIR/${model_name}_test-attn-mask-topology.log"
    reason_file="$OUT_DIR/${model_name}_test-attn-mask-reasoning.log"

    if [ -f "$topo_file" ] && [ "$(cat "$topo_file")" != "TIMEOUT" ]; then
        topo=$(parse_results "$(cat "$topo_file")")
    else
        topo="—"
    fi

    if [ -f "$reason_file" ] && [ "$(cat "$reason_file")" != "TIMEOUT" ]; then
        reason=$(parse_results "$(cat "$reason_file")")
    else
        reason="—"
    fi

    printf "║ %-22s │ %-20s │ %-20s ║\n" "$model_name" "$topo" "$reason"
done

echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Format: PASS/FAIL/INFO"
echo "  PASS  = mask correctly controls knowledge"
echo "  FAIL  = mask leakage (critical bug)"
echo "  INFO  = model-dependent (reasoning limit, not mask issue)"
echo ""
echo "Full logs: $OUT_DIR/"
