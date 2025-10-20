# Gateway Mixture-of-Experts

This directory hosts the MoE gate that fuses the existing detectors in the SDN/IoT stack.

- **Autoencoder**: `autoencoder/data/artifacts/model.pt`
- **DoS branch**: CNN `dosdet/artifacts/model_best.pt` and LSTM `Neural_LSTM/models/supervised.pt`
- **ARP branch**: CNN `arpdet/artifacts/model_best.pt` and LSTM `ARP_LSTM/models/supervised.pt`
- **Shared autoencoder**: `autoencoder/data/artifacts/model.pt` supplies window-level context for both gates

## Training the gate

```bash
cd gateway
python3 train_moe.py \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 1e-3
```

- PCAP files are discovered from the project-local `samples/` directory. Drop captures (or symlink them) there; names containing `normal` are treated as benign, while `mixed`, `attack`, `spoof`, or `dos` imply attacks for the relevant task.
- The script saves the trainable gating weights (per task) to `gateway/moe_gate.pt`. The expert checkpoints remain frozen.

### Quick smoke test

Use the optional limiter to verify the pipeline without processing every window:

```bash
python3 train_moe.py --max-batches 3
```

This performs only three optimizer steps before saving `moe_gate.pt`, which is useful for validation runs. Omit `--max-batches` for the full training pass.

### Pre-compute window features

For faster training on resource-constrained hosts, preprocess the PCAPs once to cache the window tensors:

```bash
python3 preprocess_pcaps.py \
  --tasks dos,arp \
  --batch-size 64 \
  --max-windows-per-file 120 \
  --max-file-size-mb 250 \
  --file-timeout 180
```

You’ll see a dedicated progress bar for each capture alongside the overall counter. The script streams one PCAP at a time, flushing window tensors to `gateway/cache/<tasks>/` so memory stays bounded. Long-running captures can be skipped automatically by lowering `--file-timeout` or `--max-file-size-mb`.

### Resource-constrained runs

On a 2 vCPU host with ~800 MB PCAP archives, use the new throttling flags to keep memory and CPU pressure in check:

```bash
python3 train_moe.py \
  --epochs 2 \
  --batch-size 16 \
  --num-threads 2 \
  --max-windows-per-file 150 \
  --max-total-windows 2000 \
  --gating-hidden 64 \
  --tasks dos,arp
```

- `--max-windows-per-file` and `--max-total-windows` cap the amount of data scanned per epoch (defaults are 80/1200) so large PCAPs do not overwhelm the machine.
- `--gating-hidden` shrinks the MLP width for each gate (defaults to 64) which lowers per-batch compute.
- `--tasks` lets you focus on a subset (e.g. `arp` only) when you need to reduce processing further.
- `--num-threads` pins PyTorch to at most the available CPU cores, preventing over-subscription on small hosts.
- `--max-batches` stops training after the given number of updates so you can sanity-check quickly and resume later.
- `--status-interval` emits a window-level heartbeat (default every 200 windows) so long PCAP scans do not appear frozen; set it to 0 to silence the tracker.
- `--max-file-size-mb` skips captures larger than the given size, which is useful when a single enormous PCAP would otherwise dominate the run.
- `--max-packets-per-file` truncates streaming after N packets per capture to keep CPU work bounded on noisy traces.
- `--max-packets-per-window` discards packets beyond the cap within a single window (useful when high-rate floods stall preprocessing).
- `--file-timeout` aborts a capture once the wall-clock budget is exceeded; the script logs the skip and moves on to the next file.
- `--use-cache` / `--cache-dir` control whether the trainer loads the precomputed tensors (default `auto`, which prefers the cache when present).

## Notes

- `moe_model.py` automatically locates the expert checkpoints listed above and freezes them during gate training while learning independent DoS and ARP gates.
- `train_moe.py` streams windows from `samples/` in-place or loads cached tensors when available; tune the CLI caps above to balance fidelity and runtime when working on constrained hardware. Set `--max-windows-per-file 0` if you want to process every window in a capture. When size, packet, or timeout limits trigger (either during preprocessing or streaming), the capture is skipped and annotated in the run summary so you can review which files were dropped.
- Ensure dependencies from the surrounding projects (Scapy, PyTorch, etc.) are installed; the repository’s `pyproject.toml` captures the required packages.
- Additional knobs worth knowing:
  - `--gating-hidden` shrinks/expands the gate MLP width (default 64 on this branch).
  - `--max-total-windows` sets an overall window budget per run (default 1200).
  - `--max-batches` constrains the optimiser steps per epoch for predictable run times.
  - `--status-interval` controls how often the loader reports streaming progress (0 disables it).

The training script now reports the effective batch budget derived from `--max-total-windows` and trims `--max-batches` accordingly, so status updates and the progress bar line up with the work actually performed.
