#!/bin/bash
# Pre-flight check for real experiments (run_real.py)
# Usage: bash scripts/preflight.sh

set -e
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
PASS="${GREEN}[PASS]${NC}"
FAIL="${RED}[FAIL]${NC}"
WARN="${YELLOW}[WARN]${NC}"

ERRORS=0

echo "========== Pre-flight Check =========="
echo ""

# ── 1. Python packages ──
echo "── Python 依赖 ──"
for pkg in vllm aiohttp yaml transformers numpy matplotlib; do
    if python -c "import $pkg" 2>/dev/null; then
        echo -e "  $PASS $pkg"
    else
        echo -e "  $FAIL $pkg (pip install $pkg)"
        ERRORS=$((ERRORS+1))
    fi
done

# ── 2. vLLM version ──
echo ""
echo "── vLLM 版本 ──"
VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "N/A")
echo "  vLLM version: $VLLM_VER"
if [[ "$VLLM_VER" == "N/A" ]]; then
    echo -e "  $FAIL vLLM not installed"
    ERRORS=$((ERRORS+1))
fi

# ── 3. GPU availability ──
echo ""
echo "── GPU 状态 ──"
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "  GPU count: $GPU_COUNT"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | while read line; do
        echo "  $line"
    done
    if [ "$GPU_COUNT" -lt 4 ]; then
        echo -e "  $WARN Only $GPU_COUNT GPUs (need 4 for full 2P2D)"
    else
        echo -e "  $PASS 4+ GPUs available"
    fi
else
    echo -e "  $FAIL nvidia-smi not found"
    ERRORS=$((ERRORS+1))
fi

# ── 4. Model paths ──
echo ""
echo "── 模型路径 ──"
for model_path in "/data/Qwen2.5-7B-Instruct" "/data/Llama-3.1-8B-Instruct"; do
    if [ -d "$model_path" ]; then
        echo -e "  $PASS $model_path"
    else
        echo -e "  $FAIL $model_path (not found)"
        ERRORS=$((ERRORS+1))
    fi
done

# ── 5. Config files ──
echo ""
echo "── 配置文件 ──"
for cfg in configs/models.yaml configs/workloads.yaml configs/real_scenarios.yaml configs/sim_scenarios.yaml; do
    if [ -f "$cfg" ]; then
        echo -e "  $PASS $cfg"
    else
        echo -e "  $FAIL $cfg (missing)"
        ERRORS=$((ERRORS+1))
    fi
done

# ── 6. CUDA stress kernels ──
echo ""
echo "── CUDA 压力核 ──"
if [ -f "build/cuda/libstress_interface.so" ]; then
    echo -e "  $PASS build/cuda/libstress_interface.so"
else
    echo -e "  $WARN build/cuda/libstress_interface.so (not built, run: bash scripts/00_build_cuda.sh)"
fi

# ── 7. Port availability ──
echo ""
echo "── 端口可用性 ──"
for port in 8080 8100 8101 8200 8201; do
    if ! ss -tlnp 2>/dev/null | grep -q ":$port " && ! lsof -i :$port &>/dev/null; then
        echo -e "  $PASS port $port free"
    else
        echo -e "  $FAIL port $port in use (kill the process or change port)"
        ERRORS=$((ERRORS+1))
    fi
done

# ── 8. Fonts (for plotting) ──
echo ""
echo "── 字体文件 ──"
for font in fonts/TimesNewRoman.ttf fonts/Songti.ttc; do
    if [ -f "$font" ]; then
        echo -e "  $PASS $font"
    else
        echo -e "  $WARN $font (missing, plots will use fallback font)"
    fi
done

# ── 9. Quick vLLM smoke test ──
echo ""
echo "── vLLM 快速启动测试 ──"
echo "  (skipped — run manually: CUDA_VISIBLE_DEVICES=0 python -c \"from vllm import LLM; l=LLM('/data/Qwen2.5-7B-Instruct', dtype='float16', enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.8); print('OK')\")"

# ── Summary ──
echo ""
echo "========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All checks passed. Ready to run experiments.${NC}"
else
    echo -e "${RED}$ERRORS check(s) failed. Fix the issues above before running.${NC}"
fi
