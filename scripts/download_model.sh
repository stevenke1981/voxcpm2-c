#!/usr/bin/env bash
#
# VoxCPM2 模型權重下載腳本
#
# 從 HuggingFace 下載原始 safetensors 權重，並轉換為 .vxcpm 格式。
#
# 使用方式:
#   bash scripts/download_model.sh                    # 下載並轉換 Q4
#   bash scripts/download_model.sh --quant f16        # 下載並轉換 FP16
#   bash scripts/download_model.sh --skip-convert     # 僅下載不轉換
#   bash scripts/download_model.sh --output models/custom.vxcpm
#

set -euo pipefail

# ─── 設定 ─────────────────────────────────────────────────────
HF_REPO="openbmb/VoxCPM2"
MODELS_DIR="models"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
QUANT="${1:-q4}"
OUTPUT="${MODELS_DIR}/voxcpm2-${QUANT}.vxcpm"
SKIP_CONVERT=false

# 解析參數
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quant|-q)
            QUANT="$2"
            OUTPUT="${MODELS_DIR}/voxcpm2-${QUANT}.vxcpm"
            shift 2
            ;;
        --output|-o)
            OUTPUT="$2"
            shift 2
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--quant q4|f16|f32|q8] [--output path] [--skip-convert]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ─── 前置檢查 ─────────────────────────────────────────────────
echo "=== VoxCPM2 模型下載工具 ==="
echo ""

# 檢查必要工具
command -v git &>/dev/null || { echo "❌ git 未安裝"; exit 1; }
command -v python3 &>/dev/null || { echo "❌ python3 未安裝"; exit 1; }

# 檢查 Git LFS
if ! git lfs version &>/dev/null; then
    echo "⚠ Git LFS 未安裝。大型模型檔無法正確下載。"
    echo "  安裝方式: https://git-lfs.com/"
    echo ""
    read -r -p "  是否繼續 (可能下載不完整)? [y/N] " reply
    if [[ ! "$reply" =~ ^[Yy] ]]; then
        exit 1
    fi
fi

# 檢查 Python 相依套件
echo "檢查 Python 相依套件..."
python3 -c "import torch; import safetensors" 2>/dev/null || {
    echo "安裝 Python 相依套件..."
    pip install torch safetensors numpy 2>/dev/null || {
        echo "❌ 無法安裝相依套件"
        echo "  請手動執行: pip install torch safetensors numpy"
        exit 1
    }
}

# ─── 下載 ─────────────────────────────────────────────────────
mkdir -p "${MODELS_DIR}"

if [[ -d "${MODELS_DIR}/VoxCPM2" ]]; then
    echo "📁 快取目錄存在: ${MODELS_DIR}/VoxCPM2"
    read -r -p "  重新下載? [y/N] " reply
    if [[ "$reply" =~ ^[Yy] ]]; then
        rm -rf "${MODELS_DIR}/VoxCPM2"
        echo "⬇ 從 HuggingFace 下載 VoxCPM2..."
        git lfs install
        git clone "https://huggingface.co/${HF_REPO}" "${MODELS_DIR}/VoxCPM2"
    fi
else
    echo "⬇ 從 HuggingFace 下載 VoxCPM2..."
    git lfs install
    git clone "https://huggingface.co/${HF_REPO}" "${MODELS_DIR}/VoxCPM2"
fi

echo "✅ 下載完成!"
echo ""

# ─── 權重轉換 ────────────────────────────────────────────────
if [[ "$SKIP_CONVERT" == true ]]; then
    echo "⏭ 跳過權重轉換 (--skip-convert)"
    echo "  原始權重位置: ${MODELS_DIR}/VoxCPM2/"
    echo ""
    echo "稍後可手動轉換:"
    echo "  python scripts/convert_weights.py \\"
    echo "      --input ${MODELS_DIR}/VoxCPM2 \\"
    echo "      --output ${OUTPUT} \\"
    echo "      --quant ${QUANT}"
else
    echo "🔄 轉換權重為 ${QUANT} 格式..."
    cd "${PROJECT_DIR}"
    python3 scripts/convert_weights.py \
        --input "${MODELS_DIR}/VoxCPM2" \
        --output "${OUTPUT}" \
        --quant "${QUANT}"
    
    # 驗證
    echo ""
    echo "🔍 驗證轉換結果..."
    python3 scripts/convert_weights.py --verify "${OUTPUT}"
    
    echo ""
    echo "✅ 完成!"
    echo "  權重檔案: ${OUTPUT}"
    echo "  大小: $(ls -lh "${OUTPUT}" | awk '{print $5}')"
fi
