#!/bin/bash
# Fetch the bundled `linear_spec_lora` adapter from the public HF model repo
# into `miscs/linear_spec_lora/`. The safetensors weights are gitignored
# (too large for github), so this script is needed before the first
# `eval.sh --mode linear_spec --lora` run.
#
# Usage:
#   bash scripts/fetch_bundled_lora.sh
#   bash scripts/fetch_bundled_lora.sh --model nvidia/Nemotron-Labs-Diffusion-3B   # different size
#   bash scripts/fetch_bundled_lora.sh --subfolder linear_spec_lora_v2             # different variant
set -euo pipefail

MODEL="nvidia/Nemotron-Labs-Diffusion-8B"
SUBFOLDER="linear_spec_lora"

_usage() {
    cat <<EOF
Usage: $(basename "$0") [--model HF_ID] [--subfolder NAME]

Pulls a LoRA adapter from a HuggingFace model repo into miscs/linear_spec_lora/.

Options:
  --model HF_ID       HuggingFace model id (default: $MODEL)
  --subfolder NAME    Subfolder of the model repo holding the adapter
                      (default: $SUBFOLDER)
  -h, --help          Show this help and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     MODEL="$2"; shift 2 ;;
        --subfolder) SUBFOLDER="$2"; shift 2 ;;
        -h|--help)   _usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; _usage >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/../miscs/linear_spec_lora"
mkdir -p "$DEST"

AUTH_HEADER=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    AUTH_HEADER=(-H "Authorization: Bearer $HF_TOKEN")
fi

echo "Fetching ${MODEL}/${SUBFOLDER}/ → ${DEST}/ ..."
for f in adapter_config.json adapter_model.safetensors; do
    curl -fSL "${AUTH_HEADER[@]}" \
        "https://huggingface.co/${MODEL}/resolve/main/${SUBFOLDER}/${f}" \
        -o "$DEST/$f"
done
ls -lh "$DEST"
echo "Done. Use it with: bash eval.sh --mode linear_spec --lora"
