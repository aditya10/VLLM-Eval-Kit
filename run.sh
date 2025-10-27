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


# model_id="/model-weights/Qwen2.5-VL-3B-Instruct"
# task="sg"
# dataset="gqa"
# BATCH_SIZE=16
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results" 

# model_id="/model-weights/Qwen2.5-VL-3B-Instruct"
# task="sg"
# dataset="vizwiz"
# BATCH_SIZE=16
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results" 

# model_id="/model-weights/Qwen2.5-VL-3B-Instruct"
# task="sg"
# dataset="textvqa"
# BATCH_SIZE=16
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results" 

# model_id="/model-weights/Qwen2.5-VL-3B-Instruct"
# task="sg"
# dataset="infovqa"
# BATCH_SIZE=16
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results" 

# model_id="/model-weights/Qwen2.5-VL-3B-Instruct"
# task="sg"
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

# model_id="Qwen/Qwen2.5-VL-3B-Instruct"
# task="vanilla"
# dataset="gqa"
# BATCH_SIZE=16
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results" 


#model_id="/home/mila/c/chandhos/scratch/grit_output/batch_8_dozen_vsr_qwen_add_grounded_reasoning_single_turn_think_rethink_custom_rewards/checkpoint-1100"
#model_id="/home/mila/c/chandhos/scratch/grit_output/batch_8_dozen_vsr_qwen_add_grounded_reasoning_single_turn_think_rethink_custom_rewards_vlm_llm_comparison_only/checkpoint-1250"
# model_id="Qwen/Qwen2.5-VL-3B-Instruct"
# task="vanilla"
# dataset="grit_vsr"
# BATCH_SIZE=32
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
# dataset="grit_vsr"
# BATCH_SIZE=32
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results"

model_id="/home/mila/c/chandhos/scratch/grit_output/batch_8_dozen_vsr_qwen_add_grounded_reasoning_single_turn_think_rethink_custom_rewards_vlm_llm_comparison_only/checkpoint-1250"
task="sg"
dataset="grit_vsr"
BATCH_SIZE=32
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results"

model_id="/home/mila/c/chandhos/scratch/grit_output/batch_8_dozen_vsr_qwen_add_grounded_reasoning_single_turn_think_rethink_custom_rewards_vlm_llm_comparison_v3/checkpoint-1400"
task="sg"
dataset="grit_vsr"
BATCH_SIZE=32
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results"

model_id="/home/mila/c/chandhos/scratch/grit_output/batch_8_dozen_vsr_qwen_add_grounded_reasoning_single_turn_think_rethink_custom_rewards/checkpoint-1100"
task="sg"
dataset="grit_vsr"
BATCH_SIZE=32
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results"


# model_id="Qwen/Qwen2.5-VL-3B-Instruct"
# task="vanilla"
# dataset="grit_tallyqa"
# BATCH_SIZE=32
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
# dataset="grit_tallyqa"
# BATCH_SIZE=32
# RANDOM_SEED=42

# python evaluate.py \
#     --model_id "$model_id" \
#     --task "$task" \
#     --benchmark "$dataset" \
#     --batch_size $BATCH_SIZE \
#     --random_seed $RANDOM_SEED \
#     --output_dir "./results"

model_id="/home/mila/c/chandhos/scratch/grit_output/batch_8_dozen_vsr_qwen_add_grounded_reasoning_single_turn_think_rethink_custom_rewards_vlm_llm_comparison_only/checkpoint-1250"
task="sg"
dataset="grit_tallyqa"
BATCH_SIZE=32
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results"

model_id="/home/mila/c/chandhos/scratch/grit_output/batch_8_dozen_vsr_qwen_add_grounded_reasoning_single_turn_think_rethink_custom_rewards_vlm_llm_comparison_v3/checkpoint-1400"
task="sg"
dataset="grit_tallyqa"
BATCH_SIZE=32
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results"

model_id="/home/mila/c/chandhos/scratch/grit_output/batch_8_dozen_vsr_qwen_add_grounded_reasoning_single_turn_think_rethink_custom_rewards/checkpoint-1100"
task="sg"
dataset="grit_tallyqa"
BATCH_SIZE=32
RANDOM_SEED=42

python evaluate.py \
    --model_id "$model_id" \
    --task "$task" \
    --benchmark "$dataset" \
    --batch_size $BATCH_SIZE \
    --random_seed $RANDOM_SEED \
    --output_dir "./results"