VLLM_SKIP_WARMUP=true python examples/runtime/engine/offline_batch_inference.py \
    --device cpu \
    --model-path meta-llama/Llama-3.2-1B \
    --disable-overlap-schedule \
    --dtype bfloat16 \
    --attention-backend torch_native \
    --disable-radix-cache