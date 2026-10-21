#!/bin/bash
# Budget guard for the Brev run. Every 30 min: upload logs and outputs to Google Drive and post the
# spend so far to Telegram. When the queue is done, or when the budget is reached, stop the queue
# (queue_runs.py logs the cut jobs as ABORTED), upload everything and delete this instance.
# The queue writes QUEUE_DONE only when it ran to the end; a stopped queue writes QUEUE_STOPPED.
#   experiment/budget_guard.sh <instance-name> <usd-per-hour> <usd-cap>
# Hours are counted from the machine's boot, which is when Brev starts billing. If neither the
# setup nor the queue runs for 60 min (a crash; a hand-made HOLD file pauses this), the run ends too.
cd ~/a3 || exit 1
export PATH="$HOME/.local/bin:$PATH"
NAME=$1; PRICE=$2; CAP=$3
T=$(grep ^TELEGRAM_BOT_TOKEN .env | cut -d= -f2); C=$(grep ^TELEGRAM_CHAT_ID .env | cut -d= -f2)
send() { echo "$(date -u +%FT%TZ) $1"; curl -s --max-time 20 "https://api.telegram.org/bot$T/sendMessage" \
  --data-urlencode "chat_id=$C" --data-urlencode "text=$1" >/dev/null || true; }
BOOT=$(( $(date +%s) - $(cut -d. -f1 /proc/uptime) ))
# stop 20 min before the cap: time for the last upload and the delete
STOP_AT=$(awk "BEGIN{printf \"%d\", $BOOT + ($CAP/$PRICE)*3600 - 1200}")
spent() { awk "BEGIN{printf \"%.2f\", ($(date +%s) - $BOOT)/3600*$PRICE}"; }

upload() {
  # API keys must never leave the machine
  for k in $(grep _API_KEY .env | cut -d= -f2); do
    [ -n "$k" ] && grep -rlF "$k" logs raw console_logs 2>/dev/null && { send "⚠️ API key found in outputs: upload skipped"; return 1; }
  done
  local ok=1
  for d in logs raw/rerun console_logs; do
    [ -d "$d" ] && { rclone copy "$d" "a3drive:$d" --exclude "*.pkl" --exclude "*_subset_gt.tsv" -q || ok=0; }
  done
  # partial answers, so a cut job can resume on a new machine
  rclone copy raw/rerun a3drive:resume_cache/raw/rerun --include "*.pkl" -q || ok=0
  return $((1 - ok))
}

rm -f QUEUE_DONE QUEUE_STOPPED
IDLE=0
send "🛡 budget guard on: \$$PRICE/h, cap \$$CAP, queue stops at $(date -u -d @"$STOP_AT" +%FT%TZ)"
while true; do
  for i in $(seq 30); do
    [ -f QUEUE_DONE ] && break 2
    if pgrep -f "[q]ueue_runs.py" >/dev/null || [ -f BOOTSTRAP_RUNNING ] || [ -f HOLD ]; then IDLE=0; else
      IDLE=$((IDLE + 1))
      [ $IDLE = 10 ] && send "⚠️ nothing is running on the GPU machine; it is deleted in 50 min unless the queue restarts"
      [ $IDLE -ge 60 ] && { send "⚠️ idle for 60 min: ending the run"; break 2; }
    fi
    if [ "$(date +%s)" -ge "$STOP_AT" ]; then
      send "⏰ budget reached (\$$(spent)): stopping the queue"
      echo "budget guard: cap reached" > STOP_REASON
      pkill -TERM -f "[q]ueue_runs.py"
      for j in $(seq 36); do [ -f QUEUE_STOPPED ] && break; sleep 5; done
      break 2
    fi
    sleep 60
  done
  upload && send "☁️ uploaded to Drive · spent so far ≈ \$$(spent) of \$$CAP"
done
for i in 1 2 3; do upload && { UP=ok; break; }; sleep 60; done
echo "$(date -u +%FT%TZ) run finished; spent ≈ \$$(spent); upload ${UP:-FAILED}; instance deleted next" > logs/FINISHED.txt
rclone copy logs/FINISHED.txt a3drive:logs -q
send "🏁 $(cat logs/FINISHED.txt)"
brev delete "$NAME" || send "‼️ brev delete $NAME failed: delete it by hand, it is still billing"
