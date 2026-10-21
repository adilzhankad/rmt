#!/bin/bash
# Analysis right after the queue, on the GPU machine (data and outputs are there):
# scores table, per-item scores, Kendall tau + bootstrap + figures, published-vs-ours.
cd ~/a3 || exit 1
source .venv/bin/activate
export LMUData=$HOME/a3/LMUData
set -x
python experiment/collect.py
python experiment/item_scores.py
python experiment/analyze.py --split main
python experiment/analyze.py --split pilot
python experiment/compare_published.py --split main
