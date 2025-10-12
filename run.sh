model_id="yfan1997/GRIT-20-Qwen2.5-VL-3B"
task="grit"
dataset="okvqa"
BATCH_SIZE=1
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results" \
    --limit 10