#!/bin/bash
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# ==============================================================================
# Fun-ASR-Nano 8kHz Telephone Channel Fine-tuning Script
# ==============================================================================
# This script demonstrates how to fine-tune Fun-ASR-Nano for 8kHz telephone
# audio scenarios (customer service, sales calls, VoIP, etc.)
#
# Usage:
#   bash finetune_8k_telephone.sh [stage]
#
# Stages:
#   -1: Data simulation (convert 16kHz to 8kHz telephone audio)
#    0: Data preparation (prepare train/val JSONL files)
#    1: Training/Fine-tuning
#    2: Evaluation
# ==============================================================================

# Stage control (-1: data simulation, 0: data prep, 1: train, 2: eval)
stage=${1:-1}
stop_stage=${2:-1}

# ==============================================================================
# Environment Configuration
# ==============================================================================

# GPU configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# Workspace
workspace=$(
  cd $(dirname $0)
  pwd
)
echo "Workspace: ${workspace}"

output_root="/output/funasr"
echo "Output Directory: ${output_root}"

# ==============================================================================
# Data Configuration
# ==============================================================================

# Input: High-quality 16kHz audio dataset (JSONL format)
# Expected format: {"source": "/path/to/audio.wav", "target": "transcription text", ...}
# source_train_data="/path/to/16khz/train.jsonl"
# source_val_data="/path/to/16khz/val.jsonl"
# source_train_data="/data/speech/open/data/openslr/chinese/WenetSpeech/jsonl/funasr_jsonl/train_M.jsonl"
# source_val_data="/data/speech/open/data/openslr/chinese/WenetSpeech/jsonl/funasr_jsonl/eval_dev.jsonl"
source_train_data="/output/funasr/data/yx_telecall_hybrid_300h_251225/train_labeled.jsonl"
source_val_data="/output/funasr/data/yx_telecall_hybrid_300h_251225/val_labeled.jsonl"

# Output: Simulated 8kHz telephone audio
# data_dir="${workspace}/data_8k_telephone"
data_dir="${output_root}/data/data_8k_telephone/wenetspeech"
simulated_audio_dir="${data_dir}/audio"
# train_data="${data_dir}/train_M_8k.jsonl"
# val_data="${data_dir}/eval_dev_8k.jsonl"
train_data="${data_dir}/train_telecall_8k.jsonl"
val_data="${data_dir}/val_telecall_8k.jsonl"

# ==============================================================================
# Model and Training Configuration
# ==============================================================================

# Configuration file
config="config_8k_telephone.yaml"

# Output directory for checkpoints and logs
exp_name="fun_asr_nano_8k_telephone_$(date +%Y%m%d_%H%M%S)"
output_dir="${output_root}/exp/${exp_name}"
log_file="${output_dir}/train.log"

# Initial model checkpoint (optional, for continuing training)
init_param="" # e.g., "${output_dir}/model.pt.ep10"

# DeepSpeed configuration (optional)
use_deepspeed=false
deepspeed_config="${workspace}/../../deepspeed_conf/ds_stage1.json"

# ==============================================================================
# Training Hyperparameters (can override config file)
# ==============================================================================
batch_size=4096
num_workers=32
max_epoch=20
learning_rate=0.00005 # 5e-5
accum_grad=1

# ==============================================================================
# Distributed Training Configuration
# ==============================================================================
DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26669}
"

# ==============================================================================
# Stage -1: Data Simulation (16kHz -> 8kHz Telephone Audio)
# ==============================================================================
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "===================================================="
  echo "Stage -1: Data Simulation (16kHz -> 8kHz Telephone)"
  echo "===================================================="

  mkdir -p "${data_dir}/audio_train"
  mkdir -p "${data_dir}/audio_val"

  # Check if source data exists
  if [ ! -f "${source_train_data}" ]; then
    echo "ERROR: Source training data not found: ${source_train_data}"
    echo "Please prepare your 16kHz training data in JSONL format"
    exit 1
  fi

  # Simulate training data
  echo "Simulating training data..."
  python ${workspace}/data/data_simulation.py \
    --input "${source_train_data}" \
    --output "${train_data}" \
    --output_audio_dir "${data_dir}/audio_train" \
    --audio_key "source" \
    --target_fs 8000 \
    --low_freq 300 \
    --high_freq 3400 \
    --codec_type "mu-law" \
    --snr_db_min 15 \
    --snr_db_max 25 \
    --power_line_freq 50

  # Check if source validation data exists
  if [ -f "${source_val_data}" ]; then
    echo "Simulating validation data..."
    python ${workspace}/data/data_simulation.py \
      --input "${source_val_data}" \
      --output "${val_data}" \
      --output_audio_dir "${data_dir}/audio_val" \
      --audio_key "source" \
      --target_fs 8000 \
      --low_freq 300 \
      --high_freq 3400 \
      --codec_type "mu-law" \
      --snr_db_min 15 \
      --snr_db_max 25 \
      --power_line_freq 50
  else
    echo "WARNING: Source validation data not found: ${source_val_data}"
    echo "Using training data for validation (not recommended)"
    cp "${train_data}" "${val_data}"
  fi

  echo "Data simulation completed!"
  echo "  Training data: ${train_data}"
  echo "  Validation data: ${val_data}"
fi

# ==============================================================================
# Stage 0: Data Preparation (Optional - if you have custom preprocessing)
# ==============================================================================
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "===================================================="
  echo "Stage 0: Data Preparation"
  echo "===================================================="

  # Verify data files exist
  if [ ! -f "${train_data}" ]; then
    echo "ERROR: Training data not found: ${train_data}"
    echo "Please run stage -1 first or prepare 8kHz telephone audio data"
    exit 1
  fi

  if [ ! -f "${val_data}" ]; then
    echo "ERROR: Validation data not found: ${val_data}"
    echo "Please run stage -1 first or prepare 8kHz telephone audio data"
    exit 1
  fi

  # Count samples
  train_samples=$(wc -l <"${train_data}")
  val_samples=$(wc -l <"${val_data}")

  echo "Data statistics:"
  echo "  Training samples: ${train_samples}"
  echo "  Validation samples: ${val_samples}"

  # Optional: Additional data preprocessing can be added here
  # For example: text normalization, filtering, augmentation
fi

# ==============================================================================
# Stage 1: Training/Fine-tuning
# ==============================================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "===================================================="
  echo "Stage 1: Training/Fine-tuning"
  echo "===================================================="

  # Create output directory
  mkdir -p ${output_dir}

  echo "Training configuration:"
  echo "  Config file: ${config}"
  echo "  Training data: ${train_data}"
  echo "  Validation data: ${val_data}"
  echo "  Output directory: ${output_dir}"
  echo "  Log file: ${log_file}"
  echo "  GPU count: ${gpu_num}"
  echo "  Use DeepSpeed: ${use_deepspeed}"
  echo "  Batch size: ${batch_size}"
  echo "  Learning rate: ${learning_rate}"
  echo "  Max epochs: ${max_epoch}"
  echo ""

  # Prepare init_param argument
  init_param_arg=""
  if [ -n "${init_param}" ] && [ -f "${init_param}" ]; then
    init_param_arg="++init_param=${init_param}"
    echo "Resuming from checkpoint: ${init_param}"
  fi

  # Launch training
  echo "Starting training..."
  echo $DISTRIBUTED_ARGS

  torchrun $DISTRIBUTED_ARGS \
    ${workspace}/../../../funasr/bin/train_ds.py \
    --config-path "${workspace}/conf" \
    --config-name "${config}" \
    ++train_data_set_list="${train_data}" \
    ++valid_data_set_list="${val_data}" \
    ++dataset_conf.batch_size=${batch_size} \
    ++dataset_conf.num_workers=${num_workers} \
    ++train_conf.max_epoch=${max_epoch} \
    ++train_conf.accum_grad=${accum_grad} \
    ++train_conf.use_deepspeed=${use_deepspeed} \
    ++train_conf.deepspeed_config=${deepspeed_config} \
    ++optim_conf.lr=${learning_rate} \
    ${init_param_arg} \
    ++output_dir="${output_dir}" 2>&1 | tee ${log_file}

  echo "Training completed! Checkpoints saved to: ${output_dir}"
fi

# ==============================================================================
# Stage 2: Evaluation
# ==============================================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "===================================================="
  echo "Stage 2: Evaluation"
  echo "===================================================="

  # Find the best or latest checkpoint
  if [ -n "${init_param}" ] && [ -f "${init_param}" ]; then
    ckpt_path="${init_param}"
  else
    # Use the latest checkpoint
    ckpt_path=$(ls -t ${output_dir}/model.pt.ep* 2>/dev/null | head -1)
    if [ -z "${ckpt_path}" ]; then
      echo "ERROR: No checkpoint found in ${output_dir}"
      exit 1
    fi
  fi

  ckpt_name=$(basename ${ckpt_path})
  inference_output_dir="${output_dir}/inference_${ckpt_name}"

  echo "Evaluation configuration:"
  echo "  Checkpoint: ${ckpt_path}"
  echo "  Test data: ${val_data}"
  echo "  Output directory: ${inference_output_dir}"
  echo ""

  # Run inference
  python ${workspace}/inference_8k.py \
    --model_dir "${output_dir}" \
    --checkpoint "${ckpt_path}" \
    --test_data "${val_data}" \
    --output_dir "${inference_output_dir}" \
    --device "cuda:0" \
    --batch_size 1

  echo "Evaluation completed! Results saved to: ${inference_output_dir}"
fi

echo "===================================================="
echo "All stages completed!"
echo "===================================================="
