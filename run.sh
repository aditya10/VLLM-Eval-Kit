model_id="yfan1997/GRIT-20-Qwen2.5-VL-3B"
task="grit"
dataset="vqav2"
BATCH_SIZE=16
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results" \
    --limit 30