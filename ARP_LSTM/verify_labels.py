import sys
from pathlib import Path
sys.path.append("src")
from arp_detector.data.labels import load_attack_intervals, label_windows
from arp_detector.data.structures import Window, PacketStruct

# Mock a window from pure_attack.pcap
# User said: pure_attack.pcap start-Frame 297: Jan 7, 2014, 02:56:34.433255
# My CSV has: 2014-01-07T02:56:34.433255
# Let's create a window around this time.
t0 = 1389063394.433255 # Epoch for 2014-01-07 02:56:34.433255 UTC
window = Window(
    index=0,
    start=t0,
    end=t0 + 1.0,
    packets=[PacketStruct(timestamp=t0)]
)

config_labels = type('Config', (), {'intervals_csv': 'data/arp_attack_intervals.csv', 'default_family': 'normal'})()
intervals_map = load_attack_intervals(Path("data/arp_attack_intervals.csv"), config_labels)
print("Loaded intervals map keys:", intervals_map.keys())

pcap_name = "pure_attack.pcap"
intervals = intervals_map.get(pcap_name, [])
print(f"Intervals for {pcap_name}: {len(intervals)}")
for i in intervals:
    print(f"  {i.start} -> {i.end} ({i.family})")

# Test labeling
labeled = label_windows([window], intervals, config_labels)
print(f"Label result: Attack={labeled[0].attack}, Family={labeled[0].family}")
