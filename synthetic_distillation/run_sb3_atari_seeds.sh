#!/bin/bash

conda activate spiceenv
# Array of environments to test
ENVS=("ALE/Breakout-v5" "ALE/Pong-v5" "ALE/SpaceInvaders-v5")

# Create a master directory to store the isolated logs
mkdir -p sweep_summaries/sb3_atari_st

# Define CPU ranges for 5 parallel processes
# Adjust these based on your machine's capacity
CPU_RANGES=("0-7" "8-15" "16-23" "24-31" "32-39")

for env_id in "${ENVS[@]}"; do
    # Extract short name for case matching
    env_name=$(echo $env_id | cut -d'/' -f2 | cut -d'-' -f1 | tr '[:upper:]' '[:lower:]')
    
    case "$env_name" in
        "breakout"|"pong"|"spaceinvaders")
            TIMESTEPS=5000000
            NUM_ENVS=8
            LR=2.5e-4
            EVAL_FREQ=100000
            ;;
        *)
            TIMESTEPS=1000000
            NUM_ENVS=4
            LR=3e-4
            EVAL_FREQ=50000
            ;;
    esac

    echo "================================================="
    echo "Launching parallel SB3 PPO ST-Sweep for Env: $env_id"
    echo "================================================="

    for i in {0..4}; do
        # Generate a unique seed
        SEED=$((1000 + i))
        
        CPU_RANGE=${CPU_RANGES[$i]}
        
        LOG_FILE="sweep_summaries/sb3_atari_st/${env_name}_ppo_st_seed_${SEED}_full.log"
        RES_FILE="sweep_summaries/sb3_atari_st/${env_name}_ppo_st_seed_${SEED}_final_results.txt"
        
        echo "Launching Process $((i+1)) (Seed: $SEED) on CPUs $CPU_RANGE"
        
        (
            # Run the experiment
            # Using atari_spatiotemporal config which contains a list of 5 noise types
            /opt/anaconda3/envs/spiceenv/bin/python3  main.py \
                algo=sb3_atari_ppo \
                algo.total_timesteps=$TIMESTEPS \
                algo.num_envs=$NUM_ENVS \
                algo.lr=$LR \
                +algo.eval_freq=$EVAL_FREQ \
                env=$env_name \
                seed=$SEED \
                distill=atari_spatiotemporal \
                "distill.distil_samples=[25000, 50000, 100000]" \
                "model.distil_neurons=[1.0, 0.5]" > "$LOG_FILE" 2>&1
                
            # Extract metrics
            grep -E "Training completed\.|Initializing Student|Teacher Architecture|Student Architecture|Robustness" "$LOG_FILE" > "$RES_FILE"
            
            echo "Finished $env_id | Seed: $SEED"
        ) &
    done

    # Wait for all 5 parallel seeds for the current environment to finish
    wait
    echo "All seeds for $env_id completed."
    echo ""
done

echo "Atari Spatiotemporal Sweep fully completed! Results: 'sweep_summaries/sb3_atari_st'"
