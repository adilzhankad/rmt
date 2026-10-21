#!/bin/bash
# Whole Brev run, started on a fresh instance after the repository (with .env), rclone and the
# Brev CLI were copied to it:  nohup experiment/brev_bootstrap.sh <name> <usd-per-hour> <usd-cap> &
# Sets up the three environments, downloads data and weights, then runs queue_runs.py under the
# budget guard, which deletes the instance at the end.
cd ~/a3 || exit 1
export PATH="$HOME/.local/bin:$PATH"
touch BOOTSTRAP_RUNNING
sed -i "s#^LMUData=.*#LMUData=$HOME/a3/LMUData#" .env   # .env comes from the operator's laptop
nohup experiment/budget_guard.sh "$1" "$2" "$3" > guard.log 2>&1 &
T=$(grep ^TELEGRAM_BOT_TOKEN .env | cut -d= -f2); C=$(grep ^TELEGRAM_CHAT_ID .env | cut -d= -f2)
send() { echo "$(date -u +%FT%TZ) $1"; curl -s --max-time 20 "https://api.telegram.org/bot$T/sendMessage" \
  --data-urlencode "chat_id=$C" --data-urlencode "text=$1" >/dev/null || true; }
fail() { send "❌ setup failed: $1 (see ~/a3/$1.log)"; rm -f BOOTSTRAP_RUNNING; exit 1; }

send "🔧 setup started on $1"
experiment/setup_gpu_main_env.sh > setup_main.log 2>&1 || fail setup_main
( for r in llava-hf/llava-onevision-qwen2-7b-ov-hf deepseek-ai/deepseek-vl2-small Qwen/Qwen2.5-VL-7B-Instruct \
           OpenGVLab/InternVL3-8B openbmb/MiniCPM-V-4_5 AIDC-AI/Ovis2-8B; do
    .venv/bin/hf download "$r" > /dev/null && echo "ok $r" || echo "FAILED $r"; done ) > weights.log 2>&1 &
W=$!
experiment/setup_gpu_open_env.sh > setup_open.log 2>&1 & O=$!
experiment/setup_gpu_legacy_env.sh > setup_legacy.log 2>&1 & L=$!
wait $O || fail setup_open
wait $L || fail setup_legacy
wait $W
grep -q FAILED weights.log && fail weights
send "✅ setup done: $(tail -1 setup_open.log), $(tail -1 setup_legacy.log); weights downloaded"
nohup .venv/bin/python experiment/queue_runs.py > queue.log 2>&1 &
sleep 30
rm -f BOOTSTRAP_RUNNING
