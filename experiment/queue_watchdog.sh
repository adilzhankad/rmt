#!/bin/bash
# Restarts queue_runs.py if it died without finishing (no QUEUE_DONE) and without being stopped
# (no QUEUE_STOPPED, which the budget guard waits for at the cap). A restarted queue adopts the
# run.py jobs that are still running and skips the finished ones, so nothing is redone.
#   setsid nohup experiment/queue_watchdog.sh > watchdog.log 2>&1 &
cd ~/a3 || exit 1
T=$(grep ^TELEGRAM_BOT_TOKEN .env | cut -d= -f2); C=$(grep ^TELEGRAM_CHAT_ID .env | cut -d= -f2)
send() { echo "$(date -u +%FT%TZ) $1"; curl -s --max-time 20 "https://api.telegram.org/bot$T/sendMessage" \
  --data-urlencode "chat_id=$C" --data-urlencode "text=$1" >/dev/null || true; }
while [ ! -f QUEUE_DONE ] && [ ! -f QUEUE_STOPPED ]; do
  sleep 300
  if ! pgrep -f "[q]ueue_runs.py" >/dev/null && [ ! -f QUEUE_DONE ] && [ ! -f QUEUE_STOPPED ] && [ ! -f HOLD ]; then
    send "🩹 the queue was not running: restarted it"
    setsid nohup .venv/bin/python experiment/queue_runs.py >> queue.log 2>&1 < /dev/null &
  fi
done
echo "$(date -u +%FT%TZ) watchdog ends: $(ls QUEUE_DONE QUEUE_STOPPED 2>/dev/null)"
