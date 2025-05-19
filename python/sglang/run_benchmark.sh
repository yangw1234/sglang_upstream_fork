#!/bin/bash

# Usage: ./run_benchmark.sh [--enable-torch-compile] [--model-size 8b|70b] [--batch-size N --input-len N --output-len N]
# Function to display help message
show_help() {
  echo "Usage: $0 [--enable-torch-compile] [--model-size 8b|70b] [--batch-size N --input-len N --output-len N]"
  echo "Options:"
  echo "  --enable-torch-compile   Enable compilation optimizations with torch.compile otheriwse lazy mode is used"
  echo "  --model-size 8b|70b      Specify the model size (8B or 70B) default is 8B"
  echo "  --batch-size N           Set the batch size for your benchmark"
  echo "  --input-len N            Set the input sequence length for your benchmark"
  echo "  --output-len N           Set the output sequence length for your benchmark"
  echo "  -h, --help               Show help message"
  echo " Note: if batch-size, input-len and output-len parameters are not specified, then 3 configs with following default values will be used
    config 1: 32 1024 1024
    config 2: 128 1024 1024
    config 3: 128 128 1024"
  echo " All the results will be saved in a csv file at ./results/<model_size>/<mode> directory"
  exit 0
}

# Check for --help or -h
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  show_help
fi

MODE="lazy"
EXTRA_ARGS=""
ENABLE_COMPILE=0
MODEL_SIZE="8b"

SINGLE_BS=""
SINGLE_INLEN=""
SINGLE_OUTLEN=""

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --enable-torch-compile)
            MODE="compile"
            ENABLE_COMPILE=1
            EXTRA_ARGS="--enable-torch-compile"
            ;;
        --model-size)
            shift
            MODEL_SIZE="$1"
            ;;
        --model-size=*)
            MODEL_SIZE="${1#*=}"
            ;;
        --batch-size)
            shift
            SINGLE_BS="$1"
            ;;
        --input-len)
            shift
            SINGLE_INLEN="$1"
            ;;
        --output-len)
            shift
            SINGLE_OUTLEN="$1"
            ;;
        *)
            echo "## [Error]: Invalid argument: $1"
            exit 1
            ;;
    esac
    shift
done

MODEL_SIZE="${MODEL_SIZE,,}"  # Normalize to lowercase

if [[ "$MODEL_SIZE" == "70b" ]]; then
    MODEL_PATH="/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-70B-Instruct"
    TP_SIZE="--tp-size 8"
elif [[ "$MODEL_SIZE" == "8b" ]]; then
    MODEL_PATH="/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct"
    TP_SIZE=""
else
    echo "## [Error]: Invalid model size specified: $MODEL_SIZE"
    exit 1
fi

COMMON_ARGS="--model-path $MODEL_PATH --device hpu --page-size 128 --disable-radix-cache --max-prefill-tokens 2048 --random-range-ratio 1.0 --dataset-name random --dtype bfloat16"

RESULT_DIR="./results/${MODEL_SIZE}/${MODE}"
mkdir -p "$RESULT_DIR"
CSV_FILE="${RESULT_DIR}/benchmark_summary.csv"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# CSV Header
echo "Timestamp,Model,Mode,BatchSize,InputLen,OutputLen,SuccessfulReqs,BenchmarkTime,InputTok,OutputTok,Req/s,InTok/s,OutTok/s,TotalTok/s" > "$CSV_FILE"

run_test() {
    BS=$1
    INLEN=$2
    OUTLEN=$3
    TAG=$4
    LOG_FILE="${RESULT_DIR}/${MODE}_BS${BS}_${INLEN}_${OUTLEN}_${TAG}_${TIMESTAMP}.txt"

    echo "Running: MODEL=$MODEL_SIZE, MODE=$MODE, BS=$BS, INPUT_LEN=$INLEN, OUTPUT_LEN=$OUTLEN"

    if [[ "$MODE" == "compile" ]]; then
        LAZY_MODE=0
    else
        LAZY_MODE=1
    fi

    PT_HPU_LAZY_MODE=$LAZY_MODE python3 -m sglang.bench_offline_throughput \
        $COMMON_ARGS \
        --num-prompts $BS \
        --random-input-len $INLEN \
        --random-output-len $OUTLEN \
        $EXTRA_ARGS \
        $TP_SIZE 2>&1 | tee "$LOG_FILE"

    echo "Saved log to: $LOG_FILE"

    if grep -q "Offline Throughput Benchmark Result" "$LOG_FILE"; then
        SUCCESS_REQ=$(grep "Successful requests" "$LOG_FILE" | awk -F: '{print $2}' | xargs)
        DURATION=$(grep "Benchmark duration" "$LOG_FILE" | awk -F: '{print $2}' | xargs)
        INPUT_TOK=$(grep "Total input tokens" "$LOG_FILE" | awk -F: '{print $2}' | xargs)
        OUTPUT_TOK=$(grep "Total generated tokens" "$LOG_FILE" | awk -F: '{print $2}' | xargs)
        REQ_S=$(grep "Request throughput" "$LOG_FILE" | awk -F: '{print $2}' | xargs)
        IN_TOK_S=$(grep "Input token throughput" "$LOG_FILE" | awk -F: '{print $2}' | xargs)
        OUT_TOK_S=$(grep "Output token throughput" "$LOG_FILE" | awk -F: '{print $2}' | xargs)
        TOTAL_TOK_S=$(grep "Total token throughput" "$LOG_FILE" | awk -F: '{print $2}' | xargs)

        echo "${TIMESTAMP},${MODEL_SIZE},${MODE},${BS},${INLEN},${OUTLEN},${SUCCESS_REQ},${DURATION},${INPUT_TOK},${OUTPUT_TOK},${REQ_S},${IN_TOK_S},${OUT_TOK_S},${TOTAL_TOK_S}" >> "$CSV_FILE"
    else
        echo "${TIMESTAMP},${MODEL_SIZE},${MODE},${BS},${INLEN},${OUTLEN},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$CSV_FILE"
    fi

    echo "-----------------------------------------------------"
}
if [[ -n "$SINGLE_BS" && -n "$SINGLE_INLEN" && -n "$SINGLE_OUTLEN" ]]; then
    echo "## Running single config: BS=$SINGLE_BS, INLEN=$SINGLE_INLEN, OUTLEN=$SINGLE_OUTLEN"
    run_test "$SINGLE_BS" "$SINGLE_INLEN" "$SINGLE_OUTLEN" "custom"
else
    # Run predefined benchmarks
    if [[ "$MODEL_SIZE" == "8b" ]]; then
        echo "## Running tests for 8B model..."
        run_test 32  1024 1024 "run1"
        run_test 128 1024 1024 "run2"
        run_test 128 128  1024 "run3"
    elif [[ "$MODEL_SIZE" == "70b" ]]; then
        echo "## Running tests for 70B model..."
        MODE="lazy";   run_test 32 1024 1024 "lazy_70b_BS32"
        MODE="compile"; run_test 32 1024 1024 "compile_70b_BS32"
        MODE="lazy";   run_test 128 1024 1024 "lazy_70b_BS128"
        MODE="compile"; run_test 128 1024 1024 "compile_70b_BS128"
        MODE="lazy";   run_test 128 128 1024 "lazy_70b_BS128_128_input"
        MODE="compile"; run_test 128 128 1024 "compile_70b_BS128_128_input"
    fi
fi

echo "Benchmark summary written to: $CSV_FILE"
