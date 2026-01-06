#!/bin/bash
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# ==============================================================================
# Complete Experiment Script for Fun-ASR-Nano 8kHz Telephone Fine-tuning
# ==============================================================================
# Author: Generated for three-stage fine-tuning experiment
# Date: 2026-01-05
# ==============================================================================

set -e
set -o pipefail

# ==============================================================================
# Configuration
# ==============================================================================

# GPU Configuration - can be overridden by CUDA_VISIBLE_DEVICES env var
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# Workspace - can be overridden by workspace env var
workspace="${workspace:-.}"
if [ "$workspace" = "." ]; then
  workspace=$(cd $(dirname $0) && pwd)
fi

# Output directories - can be overridden by environment variables
# Default structure: /output/funasr/{data,exp,prep}
funasr_root="${funasr_root:-/output/funasr}"
data_root="${data_root:-${funasr_root}/data}"
exp_root="${exp_root:-${funasr_root}/exp}"

# Data paths - can be overridden by environment variables
wenet_train="${wenet_train:-/data/speech/open/data/openslr/chinese/WenetSpeech/jsonl/funasr_jsonl/train_M.jsonl}"
wenet_dev="${wenet_dev:-/data/speech/open/data/openslr/chinese/WenetSpeech/jsonl/funasr_jsonl/eval_dev.jsonl}"

# Real telephone data for Stage 3
telecall_base="${telecall_base:-/data/speech/labeled/yx_telecall/sale/training_data/yx_telecall_v2_1_2025-12-22/manifests}"
telecall_train="${telecall_train:-${telecall_base}/audio_neutral.jsonl}"
telecall_dev="${telecall_dev:-${telecall_base}/audio_quiet.jsonl}"

# Processed data output
simulated_dir="${data_root}/simulated_8k_telephone"
real_data_dir="${data_root}/real_8k_telephone"
mkdir -p ${simulated_dir} ${real_data_dir}

# Stage control
stage=${1:-0}
stop_stage=${2:-3}

echo "============================================"
echo "Fun-ASR-Nano 8kHz Telephone Fine-tuning"
echo "============================================"
echo "Workspace: ${workspace}"
echo "FunASR Root: ${funasr_root}"
echo "Data: ${data_root}"
echo "Exp: ${exp_root}"
echo "GPU: ${CUDA_VISIBLE_DEVICES} (${gpu_num} GPUs)"
echo "Stage: ${stage} -> ${stop_stage}"
echo "============================================"

cd ${workspace}

# ==============================================================================
# Stage 0: Data Preparation & Simulation
# ==============================================================================
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo ""
  echo "============================================"
  echo "Stage 0: Data Preparation"
  echo "============================================"

  # # Step 0.1: Simulate telephone channel from WenetSpeech
  # echo "Step 0.1: Simulating telephone channel..."
  # echo "  Input: ${wenet_train}"
  # echo "  Output: ${simulated_dir}/train_simulated.jsonl"
  #
  # python data/data_simulation.py \
  #     --input ${wenet_train} \
  #     --output ${simulated_dir}/train_simulated.jsonl \
  #     --output_audio_dir ${simulated_dir}/audio \
  #     --audio_key source \
  #     --target_fs 8000 \
  #     --output_fs 16000 \
  #     --low_freq 300 \
  #     --high_freq 3400 \
  #     --codec_type mu-law \
  #     --snr_db_min 10 \
  #     --snr_db_max 25 \
  #     --num_workers 32
  #
  # # Simulate dev set
  # python data/data_simulation.py \
  #     --input ${wenet_dev} \
  #     --output ${simulated_dir}/dev_simulated.jsonl \
  #     --output_audio_dir ${simulated_dir}/audio \
  #     --audio_key source \
  #     --target_fs 8000 \
  #     --output_fs 16000 \
  #     --low_freq 300 \
  #     --high_freq 3400 \
  #     --codec_type mu-law \
  #     --snr_db_min 15 \
  #     --snr_db_max 25 \
  #     --num_workers 32
  #
  # # Step 0.2: Convert to FunASR training format
  # echo "Step 0.2: Converting to training format..."
  #
  # python data/prepare_training_data.py convert \
  #     --input ${simulated_dir}/train_simulated.jsonl \
  #     --output ${simulated_dir}/train_formatted.jsonl \
  #     --task_template "语音转写：" \
  #     --num_workers 32
  #
  # python data/prepare_training_data.py convert \
  #     --input ${simulated_dir}/dev_simulated.jsonl \
  #     --output ${simulated_dir}/dev_formatted.jsonl \
  #     --task_template "语音转写：" \
  #     --num_workers 32
  #
  # # Step 0.3: Prepare real telephone data (Stage 3)
  # echo "Step 0.3: Preparing real telephone data..."
  #
  # real_data_dir=${data_root}/real_8k_telephone
  # mkdir -p ${real_data_dir}/audio_upsampled
  #
  # # Filter out annotation errors and convert format
  # python data/prepare_training_data.py convert \
  #     --input ${telecall_train} \
  #     --output ${real_data_dir}/train_formatted.jsonl \
  #     --output_audio_dir ${real_data_dir}/audio_upsampled \
  #     --task_template "语音转写：" \
  #     --filter_annotation_error \
  #     --do_upsample \
  #     --target_fs 16000 \
  #     --num_workers 32
  #
  # python data/prepare_training_data.py convert \
  #     --input ${telecall_dev} \
  #     --output ${real_data_dir}/dev_formatted.jsonl \
  #     --output_audio_dir ${real_data_dir}/audio_upsampled \
  #     --task_template "语音转写：" \
  #     --filter_annotation_error \
  #     --do_upsample \
  #     --target_fs 16000 \
  #     --num_workers 32

    # Step 0.4: Validate data
    echo "Step 0.4: Validating data..."
    
    if [ -f "${simulated_dir}/train_formatted.jsonl" ]; then
        python data/prepare_training_data.py validate \
            --input ${simulated_dir}/train_formatted.jsonl \
            --sample_size 100
    fi
    
    if [ -f "${real_data_dir}/train_formatted.jsonl" ]; then
        python data/prepare_training_data.py validate \
            --input ${real_data_dir}/train_formatted.jsonl \
            --sample_size 100
    fi

  echo "Stage 0 completed!"
  echo "  Simulated train: ${simulated_dir}/train_formatted.jsonl"
  echo "  Simulated dev: ${simulated_dir}/dev_formatted.jsonl"
  echo "  Real train: ${real_data_dir}/train_formatted.jsonl"
  echo "  Real dev: ${real_data_dir}/dev_formatted.jsonl"
fi

# ==============================================================================
# Stage 1: Audio Encoder Adaptation
# ==============================================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo ""
  echo "============================================"
  echo "Stage 1: Audio Encoder Adaptation"
  echo "============================================"
  echo "Goal: Adapt encoder to 8kHz frequency spectrum"
  echo "Training: Full Encoder (~150M params)"
  echo "Frozen: Adaptor, CTC, LLM"

  config="stage1_encoder_adapt.yaml"
  exp_name="stage1_encoder_$(date +%Y%m%d_%H%M%S)"
  output_dir="${exp_root}/8k_telephone/${exp_name}"

  mkdir -p ${output_dir}

  # Check pretrained model
  ptm_checkpoint="${MODELSCOPE_CACHE:-~/.cache/modelscope}/models/FunAudioLLM/Fun-ASR-Nano-2512/model.pt"
  if [ ! -f "${ptm_checkpoint}" ]; then
    echo "Downloading pretrained model..."
    python -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-ASR-Nano-2512')"
  fi

  echo "Training Stage 1..."
  echo "  Config: ${config}"
  echo "  Init param: ${ptm_checkpoint}"
  echo "  Train data: ${simulated_dir}/train_formatted.jsonl"
  echo "  Dev data: ${simulated_dir}/dev_formatted.jsonl"
  echo "  Output: ${output_dir}"

  DISTRIBUTED_ARGS="
        --nnodes 1 \
        --nproc_per_node ${gpu_num} \
        --node_rank 0 \
        --master_addr 127.0.0.1 \
        --master_port 26670
    "

  torchrun ${DISTRIBUTED_ARGS} \
    ${workspace}/../../../funasr/bin/train_ds.py \
    --config-path "${workspace}/conf" \
    --config-name "${config}" \
    ++init_param="${ptm_checkpoint}" \
    ++train_data_set_list="${simulated_dir}/train_formatted.jsonl" \
    ++valid_data_set_list="${simulated_dir}/dev_formatted.jsonl" \
    ++output_dir="${output_dir}" 2>&1 | tee ${output_dir}/train.log

  echo "Stage 1 completed!"
  echo "  Checkpoint: ${output_dir}/model.pt.avg"

  # Save checkpoint path
  echo "${output_dir}/model.pt.avg" >${exp_root}/8k_telephone/stage1_checkpoint.txt
fi

# ==============================================================================
# Stage 2: Adapter & CTC Decoder Alignment
# ==============================================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo ""
  echo "============================================"
  echo "Stage 2: Adapter & CTC Decoder Alignment"
  echo "============================================"
  echo "Goal: Realign 8kHz features to LLM token space"
  echo "Training: Adaptor (~4M) + CTC (~8M) = ~12M params"
  echo "Frozen: Encoder (Stage 1), LLM"

  config="stage2_adapter_align.yaml"
  exp_name="stage2_adapter_$(date +%Y%m%d_%H%M%S)"
  output_dir="${exp_root}/8k_telephone/${exp_name}"

  mkdir -p ${output_dir}

  # Load Stage 1 checkpoint
  if [ -f "${exp_root}/8k_telephone/stage1_checkpoint.txt" ]; then
    stage1_ckpt=$(cat ${exp_root}/8k_telephone/stage1_checkpoint.txt)
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

  DISTRIBUTED_ARGS="
        --nnodes 1 \
        --nproc_per_node ${gpu_num} \
        --node_rank 0 \
        --master_addr 127.0.0.1 \
        --master_port 26671
    "

  torchrun ${DISTRIBUTED_ARGS} \
    ${workspace}/../../../funasr/bin/train_ds.py \
    --config-path "${workspace}/conf" \
    --config-name "${config}" \
    ++init_param="${stage1_ckpt}" \
    ++train_data_set_list="${simulated_dir}/train_formatted.jsonl" \
    ++valid_data_set_list="${simulated_dir}/dev_formatted.jsonl" \
    ++output_dir="${output_dir}" 2>&1 | tee ${output_dir}/train.log

  echo "Stage 2 completed!"
  echo "  Checkpoint: ${output_dir}/model.pt.avg"

  echo "${output_dir}/model.pt.avg" >${exp_root}/8k_telephone/stage2_checkpoint.txt
fi

# ==============================================================================
# Stage 3: LLM LoRA Domain Adaptation
# ==============================================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo ""
  echo "============================================"
  echo "Stage 3: LLM LoRA Domain Adaptation"
  echo "============================================"
  echo "Goal: Inject business terminology"
  echo "Training: LLM LoRA (~2M params)"
  echo "Frozen: Encoder, Adaptor, CTC"

  config="stage3_lora_domain.yaml"
  exp_name="stage3_lora_$(date +%Y%m%d_%H%M%S)"
  output_dir="${exp_root}/8k_telephone/${exp_name}"

  mkdir -p ${output_dir}

  # Load Stage 2 checkpoint
  if [ -f "${exp_root}/8k_telephone/stage2_checkpoint.txt" ]; then
    stage2_ckpt=$(cat ${exp_root}/8k_telephone/stage2_checkpoint.txt)
    echo "Loading Stage 2 checkpoint: ${stage2_ckpt}"
  else
    echo "ERROR: Stage 2 checkpoint not found"
    echo "Please complete Stage 2 first"
    exit 1
  fi

  real_data_dir=${data_root}/real_8k_telephone

  echo "Training Stage 3..."
  echo "  Config: ${config}"
  echo "  Init param: ${stage2_ckpt}"
  echo "  Real data: ${real_data_dir}/train_formatted.jsonl"
  echo "  Output: ${output_dir}"

  DISTRIBUTED_ARGS="
        --nnodes 1 \
        --nproc_per_node ${gpu_num} \
        --node_rank 0 \
        --master_addr 127.0.0.1 \
        --master_port 26672
    "

  torchrun ${DISTRIBUTED_ARGS} \
    ${workspace}/../../../funasr/bin/train_ds.py \
    --config-path "${workspace}/conf" \
    --config-name "${config}" \
    ++init_param="${stage2_ckpt}" \
    ++train_data_set_list="${real_data_dir}/train_formatted.jsonl" \
    ++valid_data_set_list="${real_data_dir}/dev_formatted.jsonl" \
    ++output_dir="${output_dir}" 2>&1 | tee ${output_dir}/train.log

  echo "Stage 3 completed!"

  # Save final checkpoint
  final_ckpt="${output_dir}/model.pt.avg"
  echo "${final_ckpt}" >${exp_root}/8k_telephone/final_checkpoint.txt

  echo ""
  echo "============================================"
  echo "Three-stage fine-tuning completed!"
  echo "============================================"
  echo "Final model: ${final_ckpt}"
  echo ""
  echo "Next steps:"
  echo "  1. Evaluate: python inference_8k.py --model_path ${final_ckpt}"
  echo "  2. KWER test: python evaluate_keywords.py"
  echo "============================================"
fi

echo ""
echo "Experiment completed!"
date
