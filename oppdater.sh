#!/bin/bash
# Daglig oppdatering: henter nye data fra eSett og regner ut analysen på nytt.
# Planlegg f.eks. med cron (crontab -e):
#   15 6 * * * /bin/bash "/sti/til/kortmarkeder/oppdater.sh" >> "/sti/til/kortmarkeder/data/oppdatering.log" 2>&1
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[ -x .venv/bin/python ] && PY=.venv/bin/python
echo "=== $(date '+%Y-%m-%d %H:%M') ==="
"$PY" -m kortmarkeder.lagring && "$PY" -m kortmarkeder.analyse && "$PY" -m kortmarkeder.eksport && "$PY" -m kortmarkeder.rapport
