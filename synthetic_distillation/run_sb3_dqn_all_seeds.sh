#!/bin/bash

# Array of environments to test
conda activate spiceenv

ENVS=("CartPole-v1" "Acrobot-v1" "LunarLander-v3")

# Create a master directory to store the isolated logs
mkdir -p sweep_summaries/sb3

# Define CPU ranges for 5 parallel processes
CPU_RANGES=("56-69" "70-83" "84-97" "98-111" "42-55")

for env in "${ENVS[@]}"; do
    case "$env" in
        "CartPole-v1")
            TIMESTEPS=500000
            NUM_ENVS=1
            BATCH_SIZE=32
            EPOCHS=1
            EVAL_FREQ=10000
            ;;
        "Acrobot-v1")
            TIMESTEPS=500000
            NUM_ENVS=1
            BATCH_SIZE=32
            EPOCHS=1
            EVAL_FREQ=10000
            ;;
        *)
            TIMESTEPS=500000
            NUM_ENVS=1
            BATCH_SIZE=32
            EPOCHS=1
            EVAL_FREQ=10000
            ;;
    esac

    echo "================================================="
    echo "Launching parallel SB3 DQN sweep for Env: $env | Timesteps: $TIMESTEPS"
    echo "================================================="

    for i in {0..4}; do
        # Generate a random seed based on clock time
        SEED=$(date +%s%N | cut -b10-19 | awk '{print ($1 % 1000000)}')
        # Ensure seeds are unique and varied by adding index
        SEED=$((SEED + i))
        
        CPU_RANGE=${CPU_RANGES[$i]}
        
        LOG_FILE="sweep_summaries/sb3/${env}_sb3_dqn_seed_${SEED}_full.log"
        RES_FILE="sweep_summaries/sb3/${env}_sb3_dqn_seed_${SEED}_final_results.txt"
        
        echo "Launching SB3 DQN Process $((i+1)) (Seed: $SEED) on CPUs $CPU_RANGE"
        
        (
            # Run the experiment with taskset for CPU isolation
            python main.py \
                algo=sb3_dqn \
                algo.total_timesteps=$TIMESTEPS \
                algo.num_envs=$NUM_ENVS \
                algo.batch_size=$BATCH_SIZE \
                algo.lr=1e-4 \
                algo.gamma=0.99 \
                +algo.eval_freq=$EVAL_FREQ \
                env.name=$env \
                env.type=discrete \
                seed=$SEED \
                distill=cross_entropy \
                model.neurons=64 \
                "distill.distil_samples=[5000, 10000]" \
                "model.distil_neurons=[1.0]" > "$LOG_FILE" 2>&1
                
            # Extract metrics
            grep -E "Training completed\.|Initializing Student|Teacher Architecture|Student Architecture|Robustness" "$LOG_FILE" > "$RES_FILE"
            
            echo "Finished SB3 DQN Env: $env | Seed: $SEED"
            echo "--> Extracted Robustness results saved to: $RES_FILE"
        ) &
    done

    # Wait for all 5 parallel seeds for the current environment to finish
    wait
    echo "All SB3 DQN seeds for Env: $env completed."
    echo ""
done

echo "SB3 DQN Sweep fully completed! All summaries are located in the 'sweep_summaries/sb3' directory."
