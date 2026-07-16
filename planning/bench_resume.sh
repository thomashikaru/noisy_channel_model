#!/usr/bin/env bash
# Resume the posterior-stability benchmark from wherever it stopped.
# The worker (slurm/run_nc_batch.py) is resume-safe: a fully-done config returns BEFORE loading the
# model (no GPU/JIT cost), and within a config each finished seed is reused. So re-running every phase
# in order skips all completed work and only computes what's left.
#
# Run in the background and log:
#   nohup bash planning/bench_resume.sh > planning/bench_results/resume.out 2>&1 &
set -u
cd /Users/thomasclark/mit/noisy_channel_model
echo "[$(date +%H:%M:%S)] RESUME start" >> planning/bench_results/logs/master.log
bash planning/bench_driver.sh phaseA   # only the interrupted P128 gibbs+bd config remains here
bash planning/bench_driver.sh phaseB
bash planning/bench_driver.sh phaseC
echo "[$(date +%H:%M:%S)] RESUME COMPLETE (A+B+C)" >> planning/bench_results/logs/master.log
