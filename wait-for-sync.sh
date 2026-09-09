#!/usr/bin/env bash
# Block until the codewalk dashboard syncs, then print what was said and exit.
# Run this in the background from a terminal Claude session: its completion
# notification delivers the dashboard conversation into that session's context.
#
#   wait-for-sync.sh <sync-file> [timeout-seconds]
set -u
SYNC="${1:?usage: wait-for-sync.sh <sync-file> [timeout]}"
LIMIT="${2:-3600}"
stamp() { [ -f "$SYNC" ] && stat -c %Y.%s "$SYNC" 2>/dev/null || echo none; }
BASE="$(stamp)"
END=$(( $(date +%s) + LIMIT ))
while [ "$(date +%s)" -lt "$END" ]; do
  NOW="$(stamp)"
  if [ "$NOW" != "$BASE" ] && [ -f "$SYNC" ]; then
    sleep 0.3                      # let the atomic rename settle
    echo "=== SYNCED FROM THE CODEWALK DASHBOARD ==="
    echo
    cat "$SYNC"
    exit 0
  fi
  sleep 1
done
echo "=== NO SYNC WITHIN ${LIMIT}s — still waiting in the dashboard, re-run to keep listening ==="
exit 3
