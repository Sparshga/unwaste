#!/usr/bin/env bash
# Download FreshRetailNet-50K (public, Hugging Face) into data/raw/frn50k
set -euo pipefail
mkdir -p data/raw/frn50k
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download Dingdong-Inc/FreshRetailNet-50K --repo-type dataset --local-dir data/raw/frn50k
else
  python3 -c "from huggingface_hub import snapshot_download as s; s('Dingdong-Inc/FreshRetailNet-50K', repo_type='dataset', local_dir='data/raw/frn50k')"
fi
find data/raw/frn50k -name "*.parquet" | head
