# Dynamic Mininet + Ryu + MoE Pipeline

This repository now ships with a live SDN experiment harness that stitches together:

- A Mininet topology that emulates the IoT devices listed in the research dataset.
- A Ryu controller (`dynamic_moe.controller.DynamicMoeController`) that behaves as a learning switch and forwards selected traffic windows to the unified Mixture-of-Experts IDS.
- A gateway adapter (`gateway.dynamic_moe_adapter.DynamicMoEGateway`) that wraps the frozen experts and exposes a simple `predict()` API for runtime use.
- PCAP replay helpers so you can stream historical traces while also issuing manual commands (ping, iperf, python senders) from any Mininet host.
- A dedicated runtime folder (`dynamic_moe_runtime/`) where alerts, flow decisions, packet metadata, and attack-only PCAPs are persisted.

## Dependencies

Install the SDN tooling on the host that will run Mininet:

- **Mininet** & `tcpreplay`: install via your package manager (`sudo apt install mininet tcpreplay`).
- **Ryu** controller: `pip install ryu`.
- **Scapy** (already listed in `pyproject.toml`, required for the fallback replayer).
- The unified MoE weights already reside under `gateway/`.

Ensure you can run the following commands without sudo prompts (Mininet itself still needs sudo):

```bash
which ryu-manager
which tcpreplay
```

## Running the experiment

1. **Start the Ryu controller (optional)**. The orchestration script can launch it for you, but you can also run it manually:

   ```bash
   ryu-manager dynamic_moe/controller.py
   ```

2. **Launch Mininet + replay**. The runner creates all IoT hosts, connects them to a single OpenFlow 1.3 switch, and starts the PCAP replay from the requested node. Use sudo to give Mininet the required privileges:

   ```bash
   sudo python3 -m dynamic_moe.run_dynamic_moe \
     --pcap samples/ssdp_attack.pcap \
     --replay-host h_smartthings \
     --controller-ip 127.0.0.1 \
     --ryu-app dynamic_moe.controller
   ```

   Flags of interest:

   - `--loop-pcap`: continuously replay the capture.
   - `--rate <pps>`: throttle tcpreplay to a packets-per-second rate.
   - `--replay-host <name>`: choose which IoT host injects the PCAP so the traffic inherits that device's MAC/IP identity (e.g., `h_smartthings` vs. `h_laptop`).
   - `--no-controller`: skip launching `ryu-manager` if you already started it elsewhere.
   - `--no-cli`: run headless; otherwise the Mininet CLI opens so you can drive traffic (`pingall`, `iperf`, custom scripts).

3. **Generate live traffic**. Inside the Mininet CLI you can exercise any host:

   ```bash
   mininet> h_smartthings ping -c 3 h_laptop
   mininet> iperf h_laptop h_printer
   mininet> xterm h_amazon_echo
   mininet> h_android_phone_1 python3 /tmp/custom_sender.py
   ```

   All packets (manual or replayed) are routed through the OpenFlow switch, inspected by the controller, and windowed into the MoE pipeline.
   > **Naming note:** Linux limits interface names to 15 characters, so the underlying Mininet nodes use shortened identifiers, but every device retains a friendly alias such as `h_smartthings`, `h_amazon_echo`, etc. You can continue to reference the long alias in Mininet CLI commands or when passing `--replay-host`.

   > **Controller tip:** the runner automatically exposes the repo and your user-level site-packages (e.g., Scapy installed via `pip install --user`) to `ryu-manager`. If you start Ryu manually under `sudo`, export `PYTHONPATH=/home/roots/SDN_AI_GP:$HOME/.local/lib/python3.10/site-packages` first so the controller can import its dependencies. When you intentionally keep your own `ryu-manager` session open, launch the Mininet runner with `--no-controller` (or simply rely on the runner auto-detecting that the OpenFlow port is already bound).

## Runtime outputs

All dynamic artefacts live under `dynamic_moe_runtime/` (configurable via `dynamic_moe/config.yaml`):

- `alerts.jsonl`: JSON lines for each confirmed attack (includes switch ports, device names, MoE probabilities, and expert votes).
- `moe_decisions.log`: human-readable trace of every inference window.
- `packets_meta.csv`: tabular summary per inspected window (MAC/IP pairs, protocol, score).
- `flows.csv`: records of flow programming and mitigations.
- `alerts_only.pcap`: optional PCAP containing only the frames that triggered alerts (requires scapy).

You can tailor log filenames, output directories, mitigation mode (alert-only vs. drop), and default replay host by editing `dynamic_moe/config.yaml`. The MoE adapter itself reads thresholds and checkpoint information from `gateway/config_dynamic.yaml`.

## Feature extraction & MoE API

- `dynamic_moe.feature_extractor.StreamingFeatureExtractor` mirrors the offline preprocessing pipeline (windowing, per-task scalers, gating vector assembly) so the controller can stream tensors straight into the unified MoE.
- `gateway.dynamic_moe_adapter.DynamicMoEGateway` exposes a lightweight `predict(features: dict) -> dict` interface that returns:

  ```python
  {
      "is_attack": bool,
      "attack_type": "dos" | "arp" | None,
      "score": float,
      "probabilities": {"normal": ..., "dos": ..., "arp": ...},
      "expert_votes": {"dos_cnn": 0.41, ...},
  }
  ```

  This API is reused by the Ryu controller but can also be imported in standalone scripts for quick experiments.

## Logs and troubleshooting

- Controller logs appear in the terminal running `ryu-manager`. Look for `[dynamic_moe.controller]` messages noting flow installs, detections, and mitigations.
- PCAP replay status and Mininet lifecycle events are logged by `dynamic_moe.run_dynamic_moe`.
- If `tcpreplay` is missing, the runner falls back to a Scapy-based sender (slower, but works for quick demos).
- Make sure the unified checkpoint (`gateway/unified_moe.pt`) exists; adjust `gateway/config_dynamic.yaml` if you have a custom path or thresholds.
