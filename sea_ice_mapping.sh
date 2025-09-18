cat > run_training.sh << 'EOF'
#!/bin/bash
# run_training.sh - Automated training script for sea ice classification

# Set paths
SCRIPT_DIR="/mnt/storage/Yimin/Seaice"
PARAMS_FILE="${SCRIPT_DIR}/params.json"
MAIN_SCRIPT="${SCRIPT_DIR}/new_mapping.py"
DATA_DIR="/beluga/Hack12_multi_type_seaice"
OUTPUT_DIR="${SCRIPT_DIR}/output"

# Create output directories
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/checkpoints"
mkdir -p "${OUTPUT_DIR}/results"

# Update the params.json file with the output directory
python3 -c "
import json
with open('${PARAMS_FILE}', 'r') as f:
    params = json.load(f)
params['Train_setting']['output_dir'] = '${OUTPUT_DIR}'
with open('${PARAMS_FILE}', 'w') as f:
    json.dump(params, f, indent=4)
print('✓ Updated params.json with output directory: ${OUTPUT_DIR}')
"

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Run the training script
echo "=============================================="
echo "Starting Sea Ice Classification Training"
echo "=============================================="
echo "Config file: ${PARAMS_FILE}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=============================================="

python3 "${MAIN_SCRIPT}" \
    --config "${PARAMS_FILE}" \
    --data_dir "${DATA_DIR}" \
    --log_dir "${OUTPUT_DIR}/logs" \
    --checkpoint_dir "${OUTPUT_DIR}/checkpoints" \
    --results_dir "${OUTPUT_DIR}/results" \
    --num_workers 4 \
    --gpu 0

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "=============================================="
    echo "✓ Training completed successfully!"
    echo "✓ Results saved to: ${OUTPUT_DIR}"
    echo "✓ Logs: ${OUTPUT_DIR}/logs"
    echo "✓ Checkpoints: ${OUTPUT_DIR}/checkpoints"
    echo "✓ Results: ${OUTPUT_DIR}/results"
    echo "=============================================="
else
    echo "=============================================="
    echo "✗ Training failed with exit code $?"
    echo "=============================================="
    exit 1
fi
EOF