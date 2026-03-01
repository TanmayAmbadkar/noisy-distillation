#!/bin/bash

# Array of environments to test
# conda activate spiceenv
PYTHON_PATH="/scratch1/tsa5252/anaconda3/envs/spiceenv/bin/python"

ENVS=("hopper" "halfcheetah" "ant" "humanoid")

# Create a master directory to store the isolated logs
mkdir -p sweep_summaries

# Define CPU ranges for 5 parallel processes
CPU_RANGES=("56-69" "70-83" "84-97" "98-111" "42-55")

for env in "${ENVS[@]}"; do
    if [ "$env" == "cartpole" ]; then
        TIMESTEPS=200000
    else
        TIMESTEPS=2000000
    fi

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
                algo.num_envs=4 \
                env=$env \
                seed=$SEED \
                distill=gaussian \
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
