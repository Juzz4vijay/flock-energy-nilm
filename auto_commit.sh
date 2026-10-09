#!/bin/bash
# Hourly auto-commit and push to GitHub
# Run with: bash auto_commit.sh
# Or in background: nohup bash auto_commit.sh &

REPO=/Volumes/T7/Projects/FlockEnergy

while true; do
  cd "$REPO"

  # Check if anything changed
  if ! git diff --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
    git add -A
    git commit -m "Progress update: $TIMESTAMP"
    git push origin main
    echo "[$TIMESTAMP] Committed and pushed."
  else
    echo "[$(date '+%H:%M')] No changes to commit."
  fi

  sleep 3600
done
