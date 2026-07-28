#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python preprocessing_new_position.py test_17_07_26_kmk_15/kmk_15/
python calc_darl.py
python restore.py
