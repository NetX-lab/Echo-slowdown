import json
import os
import subprocess
import torch

def get_path(command):
    try:
        return subprocess.check_output(['which', command]).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None

def update_config(file_path, updates):
    with open(file_path, 'r') as file:
        config = json.load(file)
    
    config.update(updates)
    
    with open(file_path, 'w') as file:
        json.dump(config, file, indent=4)

def main():
    current_dir = os.getcwd()
    cuda_version_installed = torch.version.cuda
    updates = {
        "cuda_visible_devices": os.getenv('CUDA_VISIBLE_DEVICES', '2,3'),
        "cuda_version_check": cuda_version_installed,
        "nsys_path": get_path('nsys'),
        "python_path": get_path('python'),
        "ncu_path": get_path('ncu')
    }

    config_files = [
        os.path.join(current_dir, 'kernel_metric/input/global_config.json'),
        os.path.join(current_dir, 'merge/input/global_config.json'),
        os.path.join(current_dir, 'slowdown_collection/input/global_config.json')
    ]

    for config_file in config_files:
        update_config(config_file, updates)

if __name__ == "__main__":
    main()