#!/usr/bin/env bash
# Wait for the running phaseA driver to finish, then run phaseB and phaseC sequentially.
cd /Users/thomasclark/mit/noisy_channel_model
while pgrep -f "bench_driver.sh phaseA" >/dev/null 2>&1; do sleep 30; done
echo "[$(date +%H:%M:%S)] phaseA process gone; starting phaseB" >> planning/bench_results/logs/master.log
bash planning/bench_driver.sh phaseB
bash planning/bench_driver.sh phaseC
echo "[$(date +%H:%M:%S)] CHAIN COMPLETE (B+C)" >> planning/bench_results/logs/master.log
