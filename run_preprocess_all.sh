#!/bin/bash
set -e

# Define paths to pcaps (assumes they are in /home/roots/SDN_AI_GP/samples or similar, 
# but previous context suggests we should rely on the user having them in a standard location 
# or passed explicitly. The user's prompt implies we should know where they are.
# Based on user's previous logs, they seem to be in `../samples/` or similar relative to project?
# Actually, the user's environment has `mixed1.pcap`, `normal1.pcap`.
# I will assume they are absolute paths or relative to common root. 
# Best guess: /home/roots/SDN_AI_GP/mixed1.pcap etc? 
# Or just rely on the fact that existing configs pointed to "samples/*.pcap".
# BUT for the script I need to be precise. 
# Let's assume they are in /home/roots/SDN_AI_GP/ as per previous context "dosdet uses mixed1.pcap and normal1.pcap".

DATA_ROOT="/home/roots/SDN_AI_GP"
DOS_PCAPS="$DATA_ROOT/mixed1.pcap $DATA_ROOT/normal1.pcap"
ARP_PCAPS="$DATA_ROOT/attack.pcap $DATA_ROOT/normal1.pcap"

# Python path helper
export PYTHONPATH=$PYTHONPATH

echo "Starting 50MB-limited preprocessing for all projects..."

# --- DOSDET Experts ---
# dosdet (CNN)
echo "[dosdet] Preprocessing..."
cd $DATA_ROOT/dosdet
python3 scripts/preprocess_50mb.py "$DATA_ROOT/*.pcap" --labels labels/labels.csv

# dosdet_16
echo "[dosdet_16] Preprocessing..."
cd $DATA_ROOT/dosdet_16
python3 scripts/preprocess_50mb.py "$DATA_ROOT/*.pcap" --labels labels/labels.csv

# dosdet_8
echo "[dosdet_8] Preprocessing..."
cd $DATA_ROOT/dosdet_8
python3 scripts/preprocess_50mb.py "$DATA_ROOT/*.pcap" --labels labels/labels.csv

# Neural_LSTM (LSTM) - accepts list of files
echo "[Neural_LSTM] Preprocessing..."
cd $DATA_ROOT/Neural_LSTM
# Ensure valid python path for internal imports
export PYTHONPATH=$DATA_ROOT/Neural_LSTM/src
python3 scripts/preprocess_50mb.py $DOS_PCAPS

# Neural_LSTM_16_mixedP
echo "[Neural_LSTM_16_mixedP] Preprocessing..."
cd $DATA_ROOT/Neural_LSTM_16_mixedP
export PYTHONPATH=$DATA_ROOT/Neural_LSTM_16_mixedP/src
python3 scripts/preprocess_50mb.py $DOS_PCAPS

# Neural_LSTM_int8
echo "[Neural_LSTM_int8] Preprocessing..."
cd $DATA_ROOT/Neural_LSTM_int8
export PYTHONPATH=$DATA_ROOT/Neural_LSTM_int8/src
python3 scripts/preprocess_50mb.py $DOS_PCAPS

# --- ARP Experts ---
# arpdet (CNN)
echo "[arpdet] Preprocessing..."
cd $DATA_ROOT/arpdet
# Note: arpdet expects glob pattern
python3 scripts/preprocess_50mb.py "$DATA_ROOT/*.pcap"

# arpdet_16
echo "[arpdet_16] Preprocessing..."
cd $DATA_ROOT/arpdet_16
python3 scripts/preprocess_50mb.py "$DATA_ROOT/*.pcap"

# arpdet_8
echo "[arpdet_8] Preprocessing..."
cd $DATA_ROOT/arpdet_8
python3 scripts/preprocess_50mb.py "$DATA_ROOT/*.pcap"

# ARP_LSTM (LSTM) - accepts glob pattern string
echo "[ARP_LSTM] Preprocessing..."
cd $DATA_ROOT/ARP_LSTM
export PYTHONPATH=$DATA_ROOT/ARP_LSTM/src
python3 scripts/preprocess_50mb.py "$DATA_ROOT/*.pcap"

# ARP_LSTM_16
echo "[ARP_LSTM_16] Preprocessing..."
cd $DATA_ROOT/ARP_LSTM_16
export PYTHONPATH=$DATA_ROOT/ARP_LSTM_16/src
python3 scripts/preprocess_50mb.py "$DATA_ROOT/*.pcap"

# ARP_LSTM_8
echo "[ARP_LSTM_8] Preprocessing..."
cd $DATA_ROOT/ARP_LSTM_8
export PYTHONPATH=$DATA_ROOT/ARP_LSTM_8/src
python3 scripts/preprocess_50mb.py "$DATA_ROOT/*.pcap"

echo "All preprocessing tasks complete."
