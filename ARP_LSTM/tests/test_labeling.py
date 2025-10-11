#tests/test_labeling.py

from pathlib import Path

from arp_detector.config import load_config
from arp_detector.data.labels import AttackInterval, label_windows
from arp_detector.data.structures import Window


def _make_window(index: int, start: float, end: float) -> Window:
    return Window(index=index, start_time=start, end_time=end, packets=[])


def test_label_windows_overlap_detection():
    config = load_config(Path("configs/config.yaml"))
    windows = [_make_window(i, float(i), float(i + 1)) for i in range(5)]
    intervals = [AttackInterval(start=1.5, end=3.2, family="arp_spoof")]
    labels = label_windows(windows, intervals, config.labels)
    attacks = [label.attack for label in labels]
    assert attacks == [0, 1, 1, 1, 0]
    families = [label.family for label in labels]
    assert families[2] == "arp_spoof"