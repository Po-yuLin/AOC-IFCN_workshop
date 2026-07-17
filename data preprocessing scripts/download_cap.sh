#!/usr/bin/env bash
# Download the PhysioNet CAP Sleep Database records used by this workshop.
# Data directory via $RBD_DATA (default: rbd_workshop/data); files go to $RBD_DATA/raw.
# Source: https://physionet.org/content/capslpdb/1.0.0/  (license: Open Data Commons Attribution)

DATA="${RBD_DATA:-rbd_workshop/data}"
RAW="$DATA/raw"
mkdir -p "$RAW"
cd "$RAW" || exit 1

BASE_URL="https://physionet.org/files/capslpdb/1.0.0"
echo "Downloading into $RAW ..."

# Healthy controls n1..n16 and RBD patients rbd1..rbd22 (.edf + staging)
for s in $(seq -f "n%g" 1 16) $(seq -f "rbd%g" 1 22); do
    echo "  $s"
    wget -q -nc "${BASE_URL}/${s}.edf"
    wget -q -nc "${BASE_URL}/${s}.edf.st"
    wget -q -nc "${BASE_URL}/${s}.txt"
done

echo "=== Done ==="
ls -lh "$RAW"/*.edf 2>/dev/null | awk '{print $5, $9}'
