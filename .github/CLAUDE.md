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

## Last Updated
- Date: 2026-01-04T06:45:30Z
