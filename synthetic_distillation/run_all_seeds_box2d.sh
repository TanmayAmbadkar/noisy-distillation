#!/bin/bash

# Array of environments to test
# conda activate spiceenv
PYTHON_PATH="/scratch1/tsa5252/anaconda3/envs/spiceenv/bin/python"

ENVS=("lunarlander" "bipedalwalker")

# Create a master directory to store the isolated logs
mkdir -p sweep_summaries

# Define CPU ranges for 5 parallel processes
CPU_RANGES=("0-6" "7-13" "14-20" "21-27" "28-34")

for env in "${ENVS[@]}"; do
    case "$env" in
        "lunarlander")
            TIMESTEPS=500000
            NUM_ENVS=4
            ROLLOUT_STEPS=1024
            MINIBATCH_SIZE=512
            EPOCHS=10
            ENTROPY_COEF=0.01
            EVAL_FREQ=10000
            ;;
        "bipedalwalker")
            TIMESTEPS=5000000
            NUM_ENVS=4
            ROLLOUT_STEPS=2048
            MINIBATCH_SIZE=1024
            EPOCHS=10
            ENTROPY_COEF=0.001
            EVAL_FREQ=100000
            ;;
        *)
            TIMESTEPS=2000000
            NUM_ENVS=4
            ROLLOUT_STEPS=2048
            MINIBATCH_SIZE=256
            EPOCHS=10
            ENTROPY_COEF=0.0
            ;;
    esac

    echo "================================================="
    echo "Launching parallel sweep for Env: $env | Timesteps: $TIMESTEPS"
    echo "================================================="

    for i in {0..4}; do
        # Generate a random seed based on clock time
        SEED=$(date +%s%N | cut -b10-19 | awk '{print ($1 % 1000000)}')
        # Ensure seeds are unique and varied by adding index
        SEED=$((SEED + i))
        
        CPU_RANGE=${CPU_RANGES[$i]}
        
        LOG_FILE="sweep_summaries/${env}_seed_${SEED}_full.log"
        RES_FILE="sweep_summaries/${env}_seed_${SEED}_final_results.txt"
        
        echo "Launching Process $((i+1)) (Seed: $SEED) on CPUs $CPU_RANGE"
        
        (
            # Run the experiment with taskset for CPU isolation
            taskset -c "$CPU_RANGE" "$PYTHON_PATH" main.py \
                algo.total_timesteps=$TIMESTEPS \
                algo.num_envs=$NUM_ENVS \
                algo.rollout_steps=$ROLLOUT_STEPS \
                algo.minibatch_size=$MINIBATCH_SIZE \
                algo.ppo_epochs=$EPOCHS \
                algo.lr=3e-4 \
                algo.anneal_lr=True \
                algo.gamma=0.99 \
                algo.gae_lambda=0.95 \
                algo.clip_eps=0.2 \
                algo.entropy_coef=$ENTROPY_COEF \
                +algo.eval_freq=$EVAL_FREQ \
                env=$env \
                seed=$SEED \
                distill=gaussian \
                model.neurons=64 \
                "distill.distil_samples=[10000, 25000, 50000, 100000]" \
                "model.distil_neurons=[1.0, 0.5, 0.25]" > "$LOG_FILE" 2>&1
                
            # Extract metrics
            grep -E "Training completed\.|Initializing Student|Teacher Architecture|Student Architecture|Robustness" "$LOG_FILE" > "$RES_FILE"
            
            echo "Finished Env: $env | Seed: $SEED"
            echo "--> Extracted Robustness results saved to: $RES_FILE"
        ) &
    done

    # Wait for all 5 parallel seeds for the current environment to finish
    wait
    echo "All seeds for Env: $env completed."
    echo ""
done

echo "Sweep fully completed! All summaries are located in the 'sweep_summaries/' directory."
