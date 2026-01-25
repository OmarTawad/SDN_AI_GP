
import datetime as dt

packet_ts = 1389048985.016165
label_str = "2014-01-06T22:56:34.433255"

# Parse label (assume UTC as per labels.py logic)
label_dt = dt.datetime.fromisoformat(label_str)
if label_dt.tzinfo is None:
    label_dt = label_dt.replace(tzinfo=dt.timezone.utc)
    
label_ts = label_dt.timestamp()

print(f"Packet TS: {packet_ts}")
print(f"Label TS:  {label_ts}")
print(f"Packet Date (UTC): {dt.datetime.fromtimestamp(packet_ts, tz=dt.timezone.utc)}")
print(f"Label Date (UTC):  {label_dt}")

diff = packet_ts - label_ts
print(f"Difference (Packet - Label): {diff:.4f} seconds")
print(f"Difference in hours: {diff/3600:.4f} hours")
