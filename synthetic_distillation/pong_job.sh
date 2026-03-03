#!/bin/bash
#PBS -l ngpus=1
#PBS -l ncpus=32
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -N pong_sweep
#PBS -M tsa5252@psu.edu
#PBS -m bea
#PBS -l mem=64g

# ══════════════════════════════════════════════════════════════════════════════
# Pong Distillation Job
# ══════════════════════════════════════════════════════════════════════════════

cd $PBS_O_WORKDIR

source ~/.bashrc2
source /scratch/tsa5252/anaconda3/etc/profile.d/conda.sh
conda activate grpo_env

# Environment variables
export HF_HOME=/scratch/tsa5252/.cache/huggingface
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /scratch/tsa5252/noisy-distillation/synthetic_distillation

# Run the python command
python -u main.py \
    env=breakout \
    algo=ppo \
    +algo.eval_freq=100000 \
    algo.num_envs=8 \
    algo.rollout_steps=128 \
    algo.clip_eps=0.1 \
    algo.ppo_epochs=4 \
    algo.lr=2.5e-4 \
    algo.anneal_lr=True \
    algo.minibatch_size=256 \
    distill=uniform \
    +distill.sampling.mode=uniform_global \
    +distill.sampling.low=0 \
    +distill.sampling.high=255 \
    algo.total_timesteps=10000000 \
    robustness.episodes=10 \
    algo.anneal_lr=False \
    distill.epochs=100 \
    'distill.distil_samples=[10000, 25000, 50000, 100000]' \
    'model.distil_neurons=[1.0, 0.5, 0.25]' \
    device="cuda" > breakout_sweep_run.log 2>&1
