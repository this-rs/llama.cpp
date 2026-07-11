#!/usr/bin/env bash
# run-qwen3.sh — Launcher for Qwen3.6-35B llama-server.
#
# ⚠️  Le modèle supporte 256k tokens, mais on N'EXPLOITE PAS naïvement cette
#     capacité. Inférence à 256k = très lent + KV-cache énorme + relecture
#     coûteuse. La stratégie : un slot de 16-24k suffisant pour un tour bien
#     cadré, et on rend les couches supérieures intelligentes (système de
#     projection compressée + memory recall scopé + tool tiering + reuse KV).
#
# Context budget par tour Vince EVG (objectif < 16k tokens / slot) :
#   - System prompt:               ~1500 tokens   (compressé à 4075c)
#   - Compressed projection:       ~3000-5000 t.  (cap activités 10, budget 10)
#   - Recent messages (6×250c):    ~500 t.
#   - Memory recall (top-K=3):     ~200-300 t.    (filtré par event:<id>)
#   - Tool definitions (à tiering):~3000-6000 t.  (55 tools = à élaguer dynamiquement)
#   - User message + headroom:     ~2000 t.
#   ─────────────────────────────────────────
#   Total : ~10-14k tokens par tour
#
# llama.cpp : ctx-size = total réservé, divisé par `parallel` (slots).
# Choix : 32768 / 2 = 16384 per slot — confortable pour le budget ci-dessus.
# --cache-reuse 256 : llama.cpp réutilise la KV-cache si le préfixe du prompt
#   correspond (énorme accélération entre tours d'une même conversation).

set -euo pipefail

MODEL="${MODEL:-/Users/triviere/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
PORT="${PORT:-9901}"
CTX_SIZE="${CTX_SIZE:-32768}"
PARALLEL="${PARALLEL:-2}"
CACHE_REUSE="${CACHE_REUSE:-256}"

echo "[run-qwen3] model       = $MODEL"
echo "[run-qwen3] port        = $PORT"
echo "[run-qwen3] ctx-size    = $CTX_SIZE (total)"
echo "[run-qwen3] parallel    = $PARALLEL (slots)"
echo "[run-qwen3] per-slot    = $((CTX_SIZE / PARALLEL)) tokens utilisables"
echo "[run-qwen3] cache-reuse = $CACHE_REUSE (KV reuse pour tours suivants)"

exec /Users/triviere/projects/ia/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --parallel "$PARALLEL" \
  --cache-reuse "$CACHE_REUSE" \
  --n-gpu-layers 999 \
  --jinja
