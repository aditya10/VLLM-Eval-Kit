# model_id="yfan1997/GRIT-20-Qwen2.5-VL-3B"
# task="grit"
# dataset="vqav2"
# BATCH_SIZE=16
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results" \
#     --limit 30

# model_id="Qwen/Qwen2.5-VL-7B-Instruct"
# task="vanilla"
# dataset="vqav2"
# BATCH_SIZE=8
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results" 

# model_id="Qwen/Qwen2.5-VL-7B-Instruct"
# task="vanilla"
# dataset="okvqa"
# BATCH_SIZE=8
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results" 


# model_id="yfan1997/GRIT-20-Qwen2.5-VL-3B"
# task="grit"
# dataset="vqav2"
# BATCH_SIZE=16
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results"

# model_id="yfan1997/GRIT-20-Qwen2.5-VL-3B"
# task="grit"
# dataset="pope"
# BATCH_SIZE=16
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results"


model_id="/model-weights/Qwen2.5-VL-3B-Instruct"
task="sg"
dataset="gqa"
BATCH_SIZE=16
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results" 

model_id="/model-weights/Qwen2.5-VL-3B-Instruct"
task="sg"
dataset="vizwiz"
BATCH_SIZE=16
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results" 

model_id="/model-weights/Qwen2.5-VL-3B-Instruct"
task="sg"
dataset="textvqa"
BATCH_SIZE=16
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results" 

model_id="/model-weights/Qwen2.5-VL-3B-Instruct"
task="sg"
dataset="infovqa"
BATCH_SIZE=16
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results" 

model_id="/model-weights/Qwen2.5-VL-3B-Instruct"
task="sg"
dataset="pope"
BATCH_SIZE=16
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results" 

model_id="/model-weights/Qwen2.5-VL-3B-Instruct"
task="sg"
dataset="okvqa"
BATCH_SIZE=16
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results" 


# model_id="checkpoint-1100"
# task="sg"
# dataset="okvqa"
# BATCH_SIZE=16
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results" 