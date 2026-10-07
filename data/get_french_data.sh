#!/usr/bin/env bash

# Fetch french.csv (Beukman 2020; distributed with Heitmeier, Chuang & Baayen 2026 at
# https://osf.io/vpdt2/).
# The OSF project states no licence for this file, so I only provide the download link
# here.

set -euo pipefail
cd "$(dirname "$0")"
curl -L --fail -o french.csv https://osf.io/download/2nm7z/
echo "french.csv: $(wc -l < french.csv) lines"
