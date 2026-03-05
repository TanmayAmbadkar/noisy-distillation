#!/bin/bash

# Array of environments to test
conda activate spiceenv

ENVS=("hopper" "halfcheetah" "humanoid" "ant")

# Create a master directory to store the isolated logs
# Experiment: Distilling from 7 Individual Global Noise Distributions
mkdir -p sweep_summaries/sb3_individual_noise

# Define CPU ranges for 5 parallel processes
CPU_RANGES=("56-69" "70-83" "84-97" "98-111" "42-55")

for env in "${ENVS[@]}"; do
    case "$env" in
        "halfcheetah")
            TIMESTEPS=2000000
            NUM_ENVS=4
            ROLLOUT_STEPS=2048
            MINIBATCH_SIZE=256
            EPOCHS=10
            EVAL_FREQ=100000
            ;;
        "hopper")
            TIMESTEPS=2000000
            NUM_ENVS=4
            ROLLOUT_STEPS=2048
            MINIBATCH_SIZE=256
            EPOCHS=10
            EVAL_FREQ=100000
            ;;
        "ant")
            TIMESTEPS=10000000
            NUM_ENVS=16
            ROLLOUT_STEPS=2048
            MINIBATCH_SIZE=4096

            EPOCHS=10
            EVAL_FREQ=100000
            ;;
        "humanoid")
            TIMESTEPS=10000000
            NUM_ENVS=32
            ROLLOUT_STEPS=1024
            MINIBATCH_SIZE=2048
            EPOCHS=10
            EVAL_FREQ=100000
            ;;
        *)
            TIMESTEPS=5000000
            NUM_ENVS=16
            ROLLOUT_STEPS=2048
            MINIBATCH_SIZE=4096
            EPOCHS=10
            EVAL_FREQ=100000
            ;;
    esac

    echo "================================================="
    echo "Launching parallel SB3 PPO sweep for Env: $env | Timesteps: $TIMESTEPS"
    echo "================================================="

    for i in {0..4}; do
        # Generate a random seed based on clock time
        SEED=$(date +%s%N | cut -b10-19 | awk '{print ($1 % 1000000)}')
        # Ensure seeds are unique and varied by adding index
        SEED=$((SEED + i))
        
        CPU_RANGE=${CPU_RANGES[$i]}
        
        LOG_FILE="sweep_summaries/sb3_individual_noise/${env}_sb3_ppo_seed_${SEED}_full.log"
        RES_FILE="sweep_summaries/sb3_individual_noise/${env}_sb3_ppo_seed_${SEED}_final_results.txt"
        
        echo "Launching SB3 PPO Process $((i+1)) (Seed: $SEED) on CPUs $CPU_RANGE"
        
        (
            # Run the experiment with taskset for CPU isolation
            /opt/anaconda3/envs/spiceenv/bin/python3 main.py \
                algo=sb3_ppo \
                algo.total_timesteps=$TIMESTEPS \
                algo.num_envs=$NUM_ENVS \
                algo.rollout_steps=$ROLLOUT_STEPS \
                algo.batch_size=$MINIBATCH_SIZE \
                algo.ppo_epochs=$EPOCHS \
                algo.lr=3e-4 \
                algo.gamma=0.99 \
                algo.gae_lambda=0.95 \
                algo.clip_eps=0.2 \
                +algo.eval_freq=$EVAL_FREQ \
                env=$env \
                seed=$SEED \
                distill=multi_noise \
                model.neurons=256 \
                "distill.distil_samples=[10000, 25000, 50000, 100000]" \
                "model.distil_neurons=[1.0, 0.5, 0.25]" > "$LOG_FILE" 2>&1
                
            # Extract metrics
            grep -E "Training completed\.|Initializing Student|Teacher Architecture|Student Architecture|Robustness" "$LOG_FILE" > "$RES_FILE"
            
            echo "Finished SB3 PPO Env: $env | Seed: $SEED"
            echo "--> Extracted Robustness results saved to: $RES_FILE"
        ) &
    done

    # Wait for all 5 parallel seeds for the current environment to finish
    wait
    echo "All SB3 PPO seeds for Env: $env completed."
    echo ""
done

echo "SB3 PPO Sweep fully completed! All summaries are located in the 'sweep_summaries/sb3_individual_noise' directory."
