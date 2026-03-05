#!/bin/bash

# Array of environments to test
# conda activate spiceenv
conda activate spiceenv

ENVS=("ant" "humanoid")

# Create a master directory to store the isolated logs
mkdir -p sweep_summaries/sac

# Define CPU ranges for 5 parallel processes
CPU_RANGES=("56-69" "70-83" "84-97" "98-111" "42-55")

for env in "${ENVS[@]}"; do
    case "$env" in
        "ant")
            TIMESTEPS=2000000
            NUM_ENVS=8
            ROLLOUT_STEPS=128
            BATCH_SIZE=256
            EVAL_FREQ=100000
            ;;
        "humanoid")
            TIMESTEPS=10000000
            NUM_ENVS=8
            ROLLOUT_STEPS=128
            BATCH_SIZE=256
            EVAL_FREQ=100000
            ;;
        *)
            TIMESTEPS=2000000
            NUM_ENVS=8
            ROLLOUT_STEPS=128
            BATCH_SIZE=256
            EVAL_FREQ=100000
            ;;
    esac

    echo "================================================="
    echo "Launching parallel SAC sweep for Env: $env | Timesteps: $TIMESTEPS"
    echo "================================================="

    for i in {0..4}; do
        # Generate a random seed based on clock time
        SEED=$(date +%s%N | cut -b10-19 | awk '{print ($1 % 1000000)}')
        # Ensure seeds are unique and varied by adding index
        SEED=$((SEED + i))
        
        CPU_RANGE=${CPU_RANGES[$i]}
        
        LOG_FILE="sweep_summaries/sac/${env}_sac_seed_${SEED}_full.log"
        RES_FILE="sweep_summaries/sac/${env}_sac_seed_${SEED}_final_results.txt"
        
        echo "Launching SAC Process $((i+1)) (Seed: $SEED) on CPUs $CPU_RANGE"
        
        (
            # Run the experiment with taskset for CPU isolation
            python main.py \
                algo=sac \
                algo.total_timesteps=$TIMESTEPS \
                algo.num_envs=$NUM_ENVS \
                algo.rollout_steps=$ROLLOUT_STEPS \
                algo.batch_size=$BATCH_SIZE \
                algo.lr=3e-4 \
                algo.gamma=0.99 \
                algo.tau=0.005 \
                algo.alpha=0.2 \
                algo.automatic_entropy_tuning=True \
                +algo.eval_freq=$EVAL_FREQ \
                env=$env \
                seed=$SEED \
                distill=gaussian \
                model.neurons=512 \
                "distill.distil_samples=[10000, 25000, 50000, 100000]" \
                "model.distil_neurons=[1.0, 0.5, 0.25]" > "$LOG_FILE" 2>&1
                
            # Extract metrics
            grep -E "Training completed\.|Initializing Student|Teacher Architecture|Student Architecture|Robustness" "$LOG_FILE" > "$RES_FILE"
            
            echo "Finished SAC Env: $env | Seed: $SEED"
            echo "--> Extracted Robustness results saved to: $RES_FILE"
        ) &
    done

    # Wait for all 5 parallel seeds for the current environment to finish
    wait
    echo "All SAC seeds for Env: $env completed."
    echo ""
done

echo "SAC Sweep fully completed! All summaries are located in the 'sweep_summaries/' directory."
