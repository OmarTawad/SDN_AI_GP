#!/usr/bin/env python3
import os
import yaml
import subprocess
from pathlib import Path

# List of projects to evaluate
PROJECTS = [
    # DOS Experts
    "dosdet", "dosdet_16", "dosdet_8",
    "Neural_LSTM", "Neural_LSTM_16_mixedP", "Neural_LSTM_int8",
    # ARP Experts
    "arpdet", "arpdet_16", "arpdet_8",
    "ARP_LSTM", "ARP_LSTM_16", "ARP_LSTM_8"
]

def run_dosdet_style(proj_dir, script_name):
    """
    Handles dosdet/arpdet style projects:
    - Config in root (config.yaml).
    - Requires override of 'split.train_val_test' to [0,0,1] to ensure all files are tested.
    """
    config_path = proj_dir / "config.yaml"
    if not config_path.exists():
        print(f"[{proj_dir.name}] SKIPPING: config.yaml not found.")
        return

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[{proj_dir.name}] ERROR loading config: {e}")
        return

    # Override split to force everything into 'test' bucket
    # Standard keys: split -> train_val_test
    if "split" in cfg and "train_val_test" in cfg["split"]:
        print(f"[{proj_dir.name}] Overriding split to [0, 0, 1] for evaluation.")
        cfg["split"]["train_val_test"] = [0.0, 0.0, 1.0]
    else:
        print(f"[{proj_dir.name}] WARNING: 'split.train_val_test' not found in config.")

    # Save temp config
    temp_config_name = "config_eval_temp.yaml"
    temp_config = proj_dir / temp_config_name
    with open(temp_config, "w") as f:
        yaml.dump(cfg, f)

    # Run evaluation command
    # Env vars to fix OMP/Torch threading issues
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    
    cmd = ["python3", script_name, "--config", temp_config_name]
    print(f"[{proj_dir.name}] Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, cwd=proj_dir, check=True, env=env)
        print(f"[{proj_dir.name}] SUCCESS")
    except subprocess.CalledProcessError:
        print(f"[{proj_dir.name}] FAILED")
    finally:
        if temp_config.exists():
            os.remove(temp_config)

def run_neural_style(proj_dir, script_name):
    """
    Handles Neural_LSTM/ARP_LSTM style projects:
    - Config usually in configs/config.yaml (handled by script defaults or we don't touch it).
    - 'test_files' empty behavior defaults to 'all remaining', so NO override needed usually.
    """
    # Verify script exists
    if not (proj_dir / script_name).exists():
         print(f"[{proj_dir.name}] SKIPPING: {script_name} not found.")
         return

    cmd = ["python3", script_name]
    print(f"[{proj_dir.name}] Running: {' '.join(cmd)}")
    
    # Needs src in PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{proj_dir}/src:{env.get('PYTHONPATH', '')}"
    
    try:
        subprocess.run(cmd, cwd=proj_dir, check=True, env=env)
        print(f"[{proj_dir.name}] SUCCESS")
    except subprocess.CalledProcessError:
        print(f"[{proj_dir.name}] FAILED")

def main():
    # Assume script is run from SDN_AI_GP root
    root = Path.cwd()
    print(f"Starting evaluation on {len(PROJECTS)} projects from {root}...")

    for proj in PROJECTS:
        proj_dir = root / proj
        if not proj_dir.exists():
            print(f"[{proj}] Directory not found, skipping.")
            continue

        # Strategy Selection
        # We distinguish based on project name or existing files.
        is_neural = "LSTM" in proj
        
        # Identify script
        script_name = None
        if not is_neural:
            # Likely dosdet/arpdet
            # Look for evaluate_*.py
            candidates = list(proj_dir.glob("evaluate_*.py"))
            if candidates:
                script_name = candidates[0].name
                run_dosdet_style(proj_dir, script_name)
            else:
                print(f"[{proj}] No evaluate_*.py script found.")
        else:
            # Neural / ARP_LSTM
            # Look for evaluate_*.py
            candidates = list(proj_dir.glob("evaluate_*.py"))
            if candidates:
                script_name = candidates[0].name
                run_neural_style(proj_dir, script_name)
            else:
                 print(f"[{proj}] No evaluate_*.py script found.")
    
    print("\nAll evaluations complete (checked).")

if __name__ == "__main__":
    main()
