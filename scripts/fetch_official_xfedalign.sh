#!/usr/bin/env bash
set -euo pipefail
# Optional reference-only checkout. The self-contained UCPA pipeline does not depend on it.
# The official repository is MIT-licensed and is linked by the ICML 2026 paper.
URL="https://github.com/dawoodwasif/xFedAlign.git"
DEST="${1:-external/xFedAlign}"
mkdir -p "$(dirname "$DEST")"
if [ -e "$DEST" ]; then echo "$DEST already exists" >&2; exit 2; fi
git clone "$URL" "$DEST"
(cd "$DEST" && git rev-parse HEAD > ../xfedalign_commit.txt)
echo "Fetched official xFedAlign into $DEST"
