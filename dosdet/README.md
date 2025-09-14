# DoS Detector for PCAPs (SSDP & General DoS)

End-to-end detector that:
1) Flags SSDP floods and other DoS types,  
2) Labels fully normal pcaps as **normal**,  
3) Labels mixed pcaps as **attack** if **any** window is suspicious.  

Learns temporal + distributional patterns; **no brittle signatures**.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
