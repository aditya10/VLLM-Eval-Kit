# python run_metrics.py --json results/Q3B/Q3B_vanilla_pope.json --metric gpt_judge --agg mean_non_zero
# python run_metrics.py --json results/Q3B/Q3B_vanilla_gqa.json --metric gpt_judge --agg mean_non_zero
# python run_metrics.py --json results/Q3B/Q3B_vanilla_okvqa.json --metric gpt_judge --agg mean_non_zero
# python run_metrics.py --json results/Q3B/Q3B_vanilla_textvqa.json --metric gpt_judge --agg mean_non_zero
# python run_metrics.py --json results/Q3B/Q3B_vanilla_vizwiz.json --metric gpt_judge --agg mean_non_zero

python run_metrics.py --json results/checkpoint-1400_sg_gqa_20251021_063332.json --metric gpt_judge --agg mean_non_zero --gptbatch batch_68f7d8f407f081908cd4b9a8cb8f5261_output.jsonl