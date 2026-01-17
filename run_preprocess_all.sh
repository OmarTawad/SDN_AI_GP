#!/bin/bash
set -e

# Default to current directory, but allow override
DATA_ROOT="${DATA_ROOT:-$(pwd)}"

echo "Looking for PCAP files in $DATA_ROOT..."

# Detect location of key files
if [ -f "$DATA_ROOT/mixed1.pcap" ]; then
    echo "Found PCAPs in $DATA_ROOT"
    PCAP_DIR="$DATA_ROOT"
elif [ -f "$DATA_ROOT/samples/mixed1.pcap" ]; then
    echo "Found PCAPs in $DATA_ROOT/samples"
    PCAP_DIR="$DATA_ROOT/samples"
else
    echo "ERROR: Could not find 'mixed1.pcap' in $DATA_ROOT or $DATA_ROOT/samples."
    echo "Please ensure PCAP files (mixed1.pcap, normal1.pcap, attack.pcap) are present."
    echo "You can set DATA_ROOT to the directory containing them."
    exit 1
fi

DOS_PCAPS="$PCAP_DIR/mixed1.pcap $PCAP_DIR/normal1.pcap"
# Adjust ARP PCAPS based on availability
if [ -f "$PCAP_DIR/attack.pcap" ]; then
    ARP_PCAPS="$PCAP_DIR/attack.pcap $PCAP_DIR/normal1.pcap"
else
    # Fallback or assume mixed1 might be used? User said "attack.pcap" for ARP.
    # If not found, warn but don't crash yet, let the tool crash if needed.
    ARP_PCAPS="$PCAP_DIR/attack.pcap $PCAP_DIR/normal1.pcap"
fi

# Python path helper
export PYTHONPATH=$PYTHONPATH

echo "Starting 50MB-limited preprocessing for all projects..."

# --- DOSDET Experts ---
# dosdet (CNN)
echo "[dosdet] Preprocessing..."
ls -l "$PCAP_DIR"/*.pcap
cd $DATA_ROOT/dosdet
python3 scripts/preprocess_50mb.py "$PCAP_DIR/*.pcap" --labels labels/labels.csv

# dosdet_16
echo "[dosdet_16] Preprocessing..."
cd $DATA_ROOT/dosdet_16
python3 scripts/preprocess_50mb.py "$PCAP_DIR/*.pcap" --labels labels/labels.csv

# dosdet_8
echo "[dosdet_8] Preprocessing..."
cd $DATA_ROOT/dosdet_8
python3 scripts/preprocess_50mb.py "$PCAP_DIR/*.pcap" --labels labels/labels.csv

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
python3 scripts/preprocess_50mb.py "$PCAP_DIR/*.pcap"

# arpdet_16
echo "[arpdet_16] Preprocessing..."
cd $DATA_ROOT/arpdet_16
python3 scripts/preprocess_50mb.py "$PCAP_DIR/*.pcap"

# arpdet_8
echo "[arpdet_8] Preprocessing..."
cd $DATA_ROOT/arpdet_8
python3 scripts/preprocess_50mb.py "$PCAP_DIR/*.pcap"

# ARP_LSTM (LSTM) - accepts glob pattern string
echo "[ARP_LSTM] Preprocessing..."
cd $DATA_ROOT/ARP_LSTM
export PYTHONPATH=$DATA_ROOT/ARP_LSTM/src
python3 scripts/preprocess_50mb.py "$PCAP_DIR/*.pcap"

# ARP_LSTM_16
echo "[ARP_LSTM_16] Preprocessing..."
cd $DATA_ROOT/ARP_LSTM_16
export PYTHONPATH=$DATA_ROOT/ARP_LSTM_16/src
python3 scripts/preprocess_50mb.py "$PCAP_DIR/*.pcap"

# ARP_LSTM_8
echo "[ARP_LSTM_8] Preprocessing..."
cd $DATA_ROOT/ARP_LSTM_8
export PYTHONPATH=$DATA_ROOT/ARP_LSTM_8/src
python3 scripts/preprocess_50mb.py "$PCAP_DIR/*.pcap"

echo "All preprocessing tasks complete."
