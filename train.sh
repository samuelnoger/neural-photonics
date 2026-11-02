#!/usr/bin/env bash
set -e

DEVICE="mps"
ARCHITECTURE="fno"
BASE_CHANNELS=32
FNO_MODES=16
FNO_LAYERS=4
EPOCHS=20
BATCH_SIZE=64
LEARNING_RATE=2e-3


# Generate dataset only if datasets/train.pt does not exist yet
if [ ! -f "datasets/train.pt" ]; then
    echo "=== Generating 2D Helmholtz Dataset ==="
    python -m data.data --n-train 2500 --n-val 300 --n-jobs -1
else
    echo "=== Skipping generation: Found existing datasets/train.pt ==="
fi

echo "=== Step 2: Training Neural Surrogate ==="
python -m train.train \
    --arch "$ARCHITECTURE" \
    --base-ch "$BASE_CHANNELS" \
    --fno-modes "$FNO_MODES" \
    --fno-layers "$FNO_LAYERS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --device "$DEVICE" \
    "$@"