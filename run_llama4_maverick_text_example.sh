PT_HPU_ENABLE_LAZY_COLLECTIVES=true PT_HPU_LAZY_MODE=1 python examples/runtime/engine/offline_batch_inference.py \
    --device hpu \
    --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct \
    --page-size 128 \
    --disable-radix-cache \
    --max-prefill-tokens 2048 \
    --tp-size 8