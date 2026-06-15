#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "GAN-DRL Cyber Deception Smart-Grid Reproducibility Pipeline"
echo "============================================================"

CONFIG_FILE="config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: config.yaml not found in the repository root."
  exit 1
fi

echo ""
echo "[1/6] Creating output directories..."
mkdir -p data/generated results results/models figures

echo ""
echo "[2/6] Generating synthetic smart-grid SCADA-IP dataset..."
python src/data_pipeline.py --config "$CONFIG_FILE"

echo ""
echo "[3/6] Training GAN and DQN models..."
python src/train.py --config "$CONFIG_FILE"

echo ""
echo "[4/6] Running baseline, ablation, and proposed-model evaluation..."
python src/evaluate.py --config "$CONFIG_FILE"

echo ""
echo "[5/6] Generating reproducible figures..."
python src/plots.py --config "$CONFIG_FILE"

echo ""
echo "[6/6] Pipeline completed successfully."
echo ""
echo "Generated outputs:"
echo "  - data/generated/"
echo "  - results/"
echo "  - figures/"
echo ""
echo "The repository is ready for archival through a DOI-minting platform after GitHub release."
