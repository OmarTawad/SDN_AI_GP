"""Dataset construction helpers for inference.


"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from gateway.data.structures.pcap import PcapInfo
from gateway.data_pipeline import MoEDataset


def build_dataloader(pcap_path: Path, batch_size: int, max_windows: int | None) -> DataLoader:
    """Construct a streaming dataloader for the provided PCAP path.

    Args:
        pcap_path: Path to the capture file.
        batch_size: Number of windows per batch emitted by the dataset.
        max_windows: Optional cap on the number of processed windows.

    Returns:
        DataLoader: Configured PyTorch dataloader yielding preprocessed windows.
    """

    info = PcapInfo(path=pcap_path, label=0)
    dataset = MoEDataset(
        files=[info],
        tasks=("dos", "arp"),
        batch_size=batch_size,
        shuffle=False,
        seed=42,
        max_windows_per_file=max_windows,
        max_total_windows=max_windows,
        status_interval=None,
        max_file_size=None,
        max_packets_per_file=None,
        file_timeout=None,
        max_packets_per_window=None,
    )
    dataset.set_window_budget(max_windows)
    dataset.set_log_fn(lambda _: None)
    return DataLoader(dataset, batch_size=None, num_workers=0)


__all__ = ["build_dataloader"]
