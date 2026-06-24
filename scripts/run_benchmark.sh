#!/usr/bin/env bash
#
# 執行 VoxCPM2-C 效能基準測試
#
# 使用方式:
#   bash scripts/run_benchmark.sh                    # 預設測試
#   bash scripts/run_benchmark.sh --gpu              # GPU 測試
#   bash scripts/run_benchmark.sh --all              # 所有量化精度
#   bash scripts/run_benchmark.sh --output results.md # 輸出至 Markdown
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
BENCH_BIN="${BUILD_DIR}/bench_rtf"
MODELS_DIR="${PROJECT_DIR}/models"
OUTPUT_FILE=""
RUN_ALL=false
USE_GPU=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) USE_GPU=true; shift ;;
        --all) RUN_ALL=true; shift ;;
        --output|-o) OUTPUT_FILE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--gpu] [--all] [--output results.md]"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# 檢查基準測試是否已編譯
if [[ ! -f "$BENCH_BIN" ]]; then
    echo "❌ Benchmark not built. Run: cmake -B build && cmake --build build"
    exit 1
fi

# 收集結果
RESULTS=()

run_bench() {
    local model="$1"
    local label="$2"
    local gpu_flag="$3"
    
    if [[ ! -f "$model" ]]; then
        echo "⚠ Model not found: $model"
        return
    fi
    
    echo ""
    echo "═══ Testing: $label ═══"
    
    local output
    output=$("$BENCH_BIN" --model "$model" $gpu_flag 2>&1)
    echo "$output"
    
    # 萃取出 RTF
    local rtf
    rtf=$(echo "$output" | grep "Mean RTF:" | awk '{print $3}')
    
    RESULTS+=("| $label | $rtf |")
}

echo "════════════════════════════════════════════════════"
echo "  VoxCPM2-C Benchmark Suite"
echo "════════════════════════════════════════════════════"
echo "  Date:    $(date)"
echo "  Host:    $(hostname)"
echo "  GPU:     $USE_GPU"
echo "════════════════════════════════════════════════════"

GPU_FLAG=""
if [[ "$USE_GPU" == true ]]; then
    GPU_FLAG="--gpu"
fi

if [[ "$RUN_ALL" == true ]]; then
    # 測試所有量化精度
    run_bench "${MODELS_DIR}/voxcpm2-f32.vxcpm" "FP32 CPU" ""
    run_bench "${MODELS_DIR}/voxcpm2-f16.vxcpm" "FP16 CPU" ""
    run_bench "${MODELS_DIR}/voxcpm2-q8.vxcpm"  "Q8 CPU"   ""
    run_bench "${MODELS_DIR}/voxcpm2-q4.vxcpm"  "Q4 CPU"   ""
    
    if [[ "$USE_GPU" == true ]]; then
        run_bench "${MODELS_DIR}/voxcpm2-f16.vxcpm" "FP16 CUDA" "--gpu"
        run_bench "${MODELS_DIR}/voxcpm2-q4.vxcpm"  "Q4 CUDA"  "--gpu"
    fi
else
    # 預設測試 Q4
    run_bench "${MODELS_DIR}/voxcpm2-q4.vxcpm" "Q4 CPU" "$GPU_FLAG"
fi

# 輸出結果表格
echo ""
echo "════════════════════════════════════════════════════"
echo "  Results Summary"
echo "════════════════════════════════════════════════════"
echo "| Configuration | Mean RTF |"
echo "|---------------|----------|"
for r in "${RESULTS[@]}"; do
    echo "$r"
done

# 寫入檔案
if [[ -n "$OUTPUT_FILE" ]]; then
    {
        echo "# VoxCPM2-C Benchmark Results"
        echo ""
        echo "Date: $(date)"
        echo "Host: $(hostname)"
        echo ""
        echo "## Summary"
        echo ""
        echo "| Configuration | Mean RTF |"
        echo "|---------------|----------|"
        for r in "${RESULTS[@]}"; do
            echo "$r"
        done
    } > "$OUTPUT_FILE"
    echo "✅ Results written to: $OUTPUT_FILE"
fi

echo ""
echo "Benchmark complete."
