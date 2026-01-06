# Claude Code Configuration

## Environment Setup

### Container Environment
This project is running in a containerized Python training environment with the following configuration:

### Python Configuration
- **Python Path**: `/opt/conda/bin/python3`
- **Type**: Conda environment (pre-configured in container)
- **Usage**: 
  - Direct execution: `/opt/conda/bin/python3 script.py`
  - Package installation: `sudo /opt/conda/bin/python3 -m pip install package_name`

### System Configuration
- **Sudo**: Passwordless sudo enabled
  - Can run: `sudo apt-get install package_name`
  - Can run: `sudo /opt/conda/bin/python3 -m pip install package_name`

### Git Configuration
- **SSH**: Not available in this environment
- **Remote Protocol**: Use HTTPS for GitHub operations
- **Origin**: `https://github.com/cppup/FunASR.git`

## Important Notes
1. Always use `/opt/conda/bin/python3` instead of system `python3` for consistency
2. Use `sudo` for package installations and system commands (no password required)
3. Use HTTPS URLs for Git operations instead of SSH
4. This is a container environment - changes outside the project directory may not persist

## Model Notes
1. modelscope model home default to $MODELSCOPE_CACHE, which is default under $HOME/cache/modelscope/;
2. huggingface model home default to $HF_HOME, which is default under $HOME/cache/huggingface/;
3. Fun-ASR-Nano PTM is under $MODELSCOPE_CACHE/models/FunAudioLLM/Fun-ASR-Nano-2512;

## Configuration Override
All script configurations can be overridden via environment variables before running:

### Directory Structure
The default directory structure uses `/output/funasr/` as the root:
```
/output/funasr/
├── data/                          # Processed datasets
│   ├── simulated_8k_telephone/   # Simulated telephone data
│   └── real_8k_telephone/        # Real telephone data
├── exp/                          # Experiment outputs
│   └── 8k_telephone/             # 8kHz telephone experiments
│       ├── stage1_encoder_*.../
│       ├── stage2_adapter_*.../
│       └── stage3_lora_*.../
└── prep/                         # Preprocessed features (optional)
```

### Example Usage
```bash
# Using default paths (/output/funasr)
bash run_experiment.sh [stage] [stop_stage]

# Override with custom paths
export funasr_root="/custom/output/funasr"
export data_root="/custom/data/path"
export exp_root="/custom/exp/path"
bash run_experiment.sh [stage] [stop_stage]

# Override GPU and data sources
export CUDA_VISIBLE_DEVICES="0,1"
export wenet_train="/path/to/train_data.jsonl"
export wenet_dev="/path/to/dev_data.jsonl"
bash run_experiment.sh [stage] [stop_stage]
```

### Supported Environment Variables
**Output Directories:**
- `funasr_root`: FunASR output root (default: /output/funasr)
- `data_root`: Processed dataset directory (default: funasr_root/data)
- `exp_root`: Experiment output directory (default: funasr_root/exp)

**Input Data Paths:**
- `wenet_train`: Training data JSONL path
- `wenet_dev`: Development data JSONL path
- `telecall_base`: Telephone call data base directory
- `telecall_train`: Telephone training data path
- `telecall_dev`: Telephone dev data path

**System Configuration:**
- `CUDA_VISIBLE_DEVICES`: GPU device IDs (default: 2,3)
- `workspace`: Working directory (default: script location)

## Last Updated
- Date: 2026-01-05T09:49:20Z
