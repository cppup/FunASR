#!/bin/bash
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# ==============================================================================
# Three-Stage Fine-tuning Script for Fun-ASR-Nano 8kHz Telephone
# ==============================================================================
# Stage 0: Data preparation (simulation + format conversion)
# Stage 1: Audio Encoder adaptation (1000h+ simulated data)
# Stage 2: Adapter & CTC alignment (1000h+ simulated data)
# Stage 3: LLM LoRA domain adaptation (50h real data)
# ==============================================================================

set -e
set -o pipefail

# Stage control
stage=${1:-1}
stop_stage=${2:-3}

# ==============================================================================
# Environment Configuration
# ==============================================================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

workspace=$(cd $(dirname $0); pwd)
echo "============================================"
echo "Workspace: ${workspace}"
echo "GPU count: ${gpu_num}"
echo "Stage: ${stage} -> ${stop_stage}"
echo "============================================"

# Output directory
output_root="${OUTPUT_ROOT:-/output/funasr}"
mkdir -p ${output_root}
echo "Output Root: ${output_root}"

# ==============================================================================
# Data Configuration
# ==============================================================================

# Raw source data (16kHz, for simulation)
raw_16k_train_data="${RAW_16K_TRAIN:-/path/to/16k_train.jsonl}"
raw_16k_val_data="${RAW_16K_VAL:-/path/to/16k_val.jsonl}"

# Simulated data output paths
simulated_data_dir="${output_root}/data/simulated_8k_telephone"
simulated_train_data="${simulated_data_dir}/train_formatted.jsonl"
simulated_val_data="${simulated_data_dir}/val_formatted.jsonl"

# Real 8kHz data (for Stage 3)
real_train_data="${REAL_TRAIN:-/path/to/real_8k_train.jsonl}"
real_val_data="${REAL_VAL:-/path/to/real_8k_val.jsonl}"

# Pretrained model
ptm_model="FunAudioLLM/Fun-ASR-Nano-2512"
ptm_checkpoint="${MODELSCOPE_CACHE:-~/.cache/modelscope}/hub/${ptm_model}/model.pt"

# ==============================================================================
# Stage 0: Data Preparation (optional)
# ==============================================================================
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "============================================"
    echo "Stage 0: Data Preparation"
    echo "============================================"
    
    # Step 0.1: Simulate telephone channel
    echo "Step 0.1: Simulating telephone channel..."
    if [ ! -f "${raw_16k_train_data}" ]; then
        echo "ERROR: Raw 16kHz data not found: ${raw_16k_train_data}"
        echo "Please set RAW_16K_TRAIN environment variable"
        exit 1
    fi
    
    python ${workspace}/data/data_simulation.py \
        --input "${raw_16k_train_data}" \
        --output "${simulated_data_dir}/train_simulated.jsonl" \
        --output_audio_dir "${simulated_data_dir}/audio" \
        --audio_key source \
        --target_fs 8000 \
        --output_fs 16000 \
        --low_freq 300 \
        --high_freq 3400 \
        --codec_type mu-law \
        --snr_db_min 10 \
        --snr_db_max 25 \
        --num_workers 64
    
    # Step 0.2: Convert to FunASR training format
    echo "Step 0.2: Converting to training format..."
    python ${workspace}/data/prepare_training_data.py convert \
        --input "${simulated_data_dir}/train_simulated.jsonl" \
        --output "${simulated_train_data}" \
        --task_template "语音转写：" \
        --num_workers 32
    
    # Step 0.3: Validate data
    echo "Step 0.3: Validating data..."
    python ${workspace}/data/prepare_training_data.py validate \
        --input "${simulated_train_data}"
    
    echo "Stage 0 completed. Data saved to: ${simulated_data_dir}"
fi

# ==============================================================================
# Stage 1: Audio Encoder Adaptation
# ==============================================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "============================================"
    echo "Stage 1: Audio Encoder Adaptation"
    echo "============================================"
    echo "Goal: Adapt encoder to 8kHz frequency spectrum"
    echo "Training: Full Encoder (~150M params)"
    echo "Frozen: Adaptor, CTC, LLM"
    echo ""
    
    config="stage1_encoder_adapt.yaml"
    exp_name="stage1_encoder_$(date +%Y%m%d_%H%M%S)"
    output_dir="${output_root}/exp/8k_telephone/${exp_name}"
    log_file="${output_dir}/train.log"
    
    mkdir -p ${output_dir}
    
    # Check data
    if [ ! -f "${simulated_train_data}" ]; then
        echo "ERROR: Training data not found: ${simulated_train_data}"
        echo "Please run Stage 0 first, or set correct data path"
        exit 1
    fi
    
    # Check pretrained model
    if [ ! -f "${ptm_checkpoint}" ]; then
        echo "Downloading pretrained model..."
        python -c "from modelscope import snapshot_download; snapshot_download('${ptm_model}')"
    fi
    
    echo "Training Stage 1..."
    echo "  Config: ${config}"
    echo "  Init param: ${ptm_checkpoint}"
    echo "  Train data: ${simulated_train_data}"
    echo "  Output: ${output_dir}"
    echo ""
    
    DISTRIBUTED_ARGS="
        --nnodes ${WORLD_SIZE:-1} \
        --nproc_per_node $gpu_num \
        --node_rank ${RANK:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-26670}
    "
    
    torchrun $DISTRIBUTED_ARGS \
        ${workspace}/../../../funasr/bin/train_ds.py \
        --config-path "${workspace}/conf" \
        --config-name "${config}" \
        ++init_param="${ptm_checkpoint}" \
        ++train_data_set_list="${simulated_train_data}" \
        ++valid_data_set_list="${simulated_val_data}" \
        ++output_dir="${output_dir}" 2>&1 | tee ${log_file}
    
    echo ""
    echo "Stage 1 completed!"
    echo "  Checkpoint: ${output_dir}/model.pt.avg"
    
    # Save checkpoint path
    mkdir -p ${output_root}/exp/8k_telephone
    echo "${output_dir}/model.pt.avg" > ${output_root}/exp/8k_telephone/stage1_checkpoint.txt
fi

# ==============================================================================
# Stage 2: Adapter & CTC Decoder Alignment
# ==============================================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "============================================"
    echo "Stage 2: Adapter & CTC Decoder Alignment"
    echo "============================================"
    echo "Goal: Realign 8kHz features to LLM token space"
    echo "Training: Adaptor (~4M) + CTC (~8M) = ~12M params"
    echo "Frozen: Encoder (Stage 1), LLM"
    echo ""
    
    config="stage2_adapter_align.yaml"
    exp_name="stage2_adapter_$(date +%Y%m%d_%H%M%S)"
    output_dir="${output_root}/exp/8k_telephone/${exp_name}"
    log_file="${output_dir}/train.log"
    
    mkdir -p ${output_dir}
    
    # Load Stage 1 checkpoint
    if [ -f "${output_root}/exp/8k_telephone/stage1_checkpoint.txt" ]; then
        stage1_ckpt=$(cat ${output_root}/exp/8k_telephone/stage1_checkpoint.txt)
        echo "Loading Stage 1 checkpoint: ${stage1_ckpt}"
    else
        echo "ERROR: Stage 1 checkpoint not found"
        echo "Please complete Stage 1 first"
        exit 1
    fi
    
    echo "Training Stage 2..."
    echo "  Config: ${config}"
    echo "  Init param: ${stage1_ckpt}"
    echo "  Output: ${output_dir}"
    echo ""
    
    DISTRIBUTED_ARGS="
        --nnodes ${WORLD_SIZE:-1} \
        --nproc_per_node $gpu_num \
        --node_rank ${RANK:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-26671}
    "
    
    torchrun $DISTRIBUTED_ARGS \
        ${workspace}/../../../funasr/bin/train_ds.py \
        --config-path "${workspace}/conf" \
        --config-name "${config}" \
        ++init_param="${stage1_ckpt}" \
        ++train_data_set_list="${simulated_train_data}" \
        ++valid_data_set_list="${simulated_val_data}" \
        ++output_dir="${output_dir}" 2>&1 | tee ${log_file}
    
    echo ""
    echo "Stage 2 completed!"
    echo "  Checkpoint: ${output_dir}/model.pt.avg"
    
    echo "${output_dir}/model.pt.avg" > ${output_root}/exp/8k_telephone/stage2_checkpoint.txt
fi

# ==============================================================================
# Stage 3: LLM LoRA Domain Adaptation
# ==============================================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "============================================"
    echo "Stage 3: LLM LoRA Domain Adaptation"
    echo "============================================"
    echo "Goal: Inject business terminology"
    echo "Training: LLM LoRA (~2M params)"
    echo "Frozen: Encoder, Adaptor, CTC"
    echo ""
    
    config="stage3_lora_domain.yaml"
    exp_name="stage3_lora_$(date +%Y%m%d_%H%M%S)"
    output_dir="${output_root}/exp/8k_telephone/${exp_name}"
    log_file="${output_dir}/train.log"
    
    mkdir -p ${output_dir}
    
    # Load Stage 2 checkpoint
    if [ -f "${output_root}/exp/8k_telephone/stage2_checkpoint.txt" ]; then
        stage2_ckpt=$(cat ${output_root}/exp/8k_telephone/stage2_checkpoint.txt)
        echo "Loading Stage 2 checkpoint: ${stage2_ckpt}"
    else
        echo "ERROR: Stage 2 checkpoint not found"
        echo "Please complete Stage 2 first"
        exit 1
    fi
    
    # Check real data
    if [ ! -f "${real_train_data}" ]; then
        echo "ERROR: Real training data not found: ${real_train_data}"
        echo "Please set REAL_TRAIN environment variable"
        exit 1
    fi
    
    echo "Training Stage 3..."
    echo "  Config: ${config}"
    echo "  Init param: ${stage2_ckpt}"
    echo "  Real data: ${real_train_data}"
    echo "  Output: ${output_dir}"
    echo ""
    
    DISTRIBUTED_ARGS="
        --nnodes ${WORLD_SIZE:-1} \
        --nproc_per_node $gpu_num \
        --node_rank ${RANK:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-26672}
    "
    
    torchrun $DISTRIBUTED_ARGS \
        ${workspace}/../../../funasr/bin/train_ds.py \
        --config-path "${workspace}/conf" \
        --config-name "${config}" \
        ++init_param="${stage2_ckpt}" \
        ++train_data_set_list="${real_train_data}" \
        ++valid_data_set_list="${real_val_data}" \
        ++output_dir="${output_dir}" 2>&1 | tee ${log_file}
    
    echo ""
    echo "Stage 3 completed!"
    
    # Save final checkpoint
    final_ckpt="${output_dir}/model.pt.avg"
    echo "${final_ckpt}" > ${output_root}/exp/8k_telephone/final_checkpoint.txt
    
    echo "============================================"
    echo "Three-stage fine-tuning completed!"
    echo "============================================"
    echo "Final model: ${final_ckpt}"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate: python inference_8k.py --model_path ${final_ckpt}"
    echo "  2. KWER test: python evaluate_keywords.py --results results.jsonl --keywords keywords_example.txt"
    echo "============================================"
fi
