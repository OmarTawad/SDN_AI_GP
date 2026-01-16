#!/usr/bin/env python3
import subprocess
import os
import sys

def run_command(cmd, cwd=None, env=None):
    """Run a shell command."""
    print(f"[INFO] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}")
        # We don't exit here to allow other steps to try running
        # sys.exit(1) 

def main():
    # Base directory assumed to be SDN_AI_GP
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # 1. ARP CNN (arpdet)
    print("\n--- Preprocessing ARP CNN (arpdet) ---")
    arpdet_config = os.path.join(base_dir, "arpdet", "config.yaml")
    # Using the files specified by user
    arp_pcaps = [
        os.path.join(base_dir, "samples", "attack.pcap"),
        os.path.join(base_dir, "samples", "normal1.pcap")
    ]
    
    # Existing arpdet preprocess script handles glob, so we pass individual files or a pattern.
    # The script accepts --pcaps as a glob override.
    # We will run it for each file or construct a glob if possible, but the script takes a single string arg for glob usually?
    # Let's check imports. arpdet/data/preprocess.py takes nargs='+' for pcaps.
    
    arpdet_cmd = [
        "python3", "arpdet/data/preprocess.py",
        "--config", arpdet_config,
        "--limit", "500",
        "--pcaps"
    ] + arp_pcaps
    
    # Run from SDN_AI_GP root so python path works for 'arpdet' imports? 
    # arpdet code uses 'from data.pcap_reader...' which implies it expects to be run relative to where 'data' package is,
    # OR it runs as module. 
    # Looking at `arpdet/data/preprocess.py`: `from data.pcap_reader` 
    # If we run `python3 arpdet/data/preprocess.py`, then `sys.path[0]` is `arpdet/data`.
    # But `from data` would fail if we are inside data. 
    # Wait, `arpdet/data/preprocess.py` has `from data.pcap_reader import ...`. 
    # This implies it is run from `arpdet` directory? Or `data` is a package in `arpdet`?
    # Actually, if I run from `SDN_AI_GP`, I need `PYTHONPATH` to include `arpdet`? 
    # Let's assume the user runs it from SDN_AI_GP.
    # The safest bet for `arpdet` legacy code is often running from the module root.
    
    # Run from arpdet directory to match expected import structure (if configured that way)
    # We add the current directory to PYTHONPATH so 'from data.pcap_reader' works.
    env_arp = os.environ.copy()
    env_arp["PYTHONPATH"] = os.path.join(base_dir, "arpdet") + ":" + env_arp.get("PYTHONPATH", "")
    
    run_command(
        ["python3", "data/preprocess.py", "--config", "config.yaml", "--limit", "500", "--pcaps"] + [os.path.relpath(p, os.path.join(base_dir, "arpdet")) for p in arp_pcaps],
        cwd=os.path.join(base_dir, "arpdet"),
        env=env_arp
    )

    # 2. DOS CNN (dosdet)
    print("\n--- Preprocessing DOS CNN (dosdet) ---")
    dosdet_config = os.path.join(base_dir, "dosdet", "config.yaml")
    dos_pcaps = [
        os.path.join(base_dir, "dosdet", "samples", "mixed1.pcap"),
        os.path.join(base_dir, "dosdet", "samples", "normal1.pcap")
    ]
    
    env_dos_cnn = os.environ.copy()
    env_dos_cnn["PYTHONPATH"] = os.path.join(base_dir, "dosdet") + ":" + env_dos_cnn.get("PYTHONPATH", "")

    run_command(
        ["python3", "data/preprocess.py", "--config", "config.yaml", "--limit", "500", "--pcaps"] + [os.path.relpath(p, os.path.join(base_dir, "dosdet")) for p in dos_pcaps],
        cwd=os.path.join(base_dir, "dosdet"),
        env=env_dos_cnn
    )

    # 3. ARP LSTM
    print("\n--- Preprocessing ARP LSTM ---")
    # Uses 'src.arp_detector.cli'. Needs PYTHONPATH to include 'src'.
    # Running from ARP_LSTM directory seems appropriate.
    arp_lstm_dir = os.path.join(base_dir, "ARP_LSTM")
    
    # construct pcap pattern or list. The CLI accepts a glob pattern string.
    # It might take a single argument for pattern. 
    # cli.py: `pcaps: str = typer.Argument(..., help="Glob pattern for PCAP files")`
    # It seems to take ONE argument which is a glob string.
    # If the files are in different directories or specific files, glob might be tricky if they don't share a common pattern excluding others.
    # However, the user said "samples/attack.pcap" and "samples/normal1.pcap". 
    # We can pass specific paths if we run it twice or if the glob can catch them. 
    # Typer Argument usually takes one string.
    # I will run it once per file to be safe and ensure the limit applies to each file (limit arg is "per file" in the code).
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(arp_lstm_dir, "src") + ":" + env.get("PYTHONPATH", "")

    for pcap in arp_pcaps:
        run_command(
            [
                "python3", "-m", "arp_detector.cli", "extract-features",
                pcap, # Just passing the file path as the 'glob'
                "--config-path", "configs/config.yaml",
                "--limit", "500"
            ],
            cwd=arp_lstm_dir,
            env=env
        )

    # 4. DOS LSTM (Neural_LSTM)
    print("\n--- Preprocessing DOS LSTM (Neural_LSTM) ---")
    neural_lstm_dir = os.path.join(base_dir, "Neural_LSTM")
    env_dos = os.environ.copy()
    env_dos["PYTHONPATH"] = os.path.join(neural_lstm_dir, "src") + ":" + env_dos.get("PYTHONPATH", "")
    
    for pcap in dos_pcaps:
        run_command(
            [
                "python3", "-m", "dos_detector.cli", "extract-features",
                pcap,
                "--config-path", "configs/config.yaml",
                "--limit", "500"
            ],
            cwd=neural_lstm_dir,
            env=env_dos
        )

    print("\n[INFO] Limited preprocessing tasks completed.")

if __name__ == "__main__":
    main()
