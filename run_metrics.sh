python run_metrics.py --json results/Q3B/Q3B_vanilla_pope.json --metric gpt_judge --agg mean_non_zero
python run_metrics.py --json results/Q3B/Q3B_vanilla_gqa.json --metric gpt_judge --agg mean_non_zero
python run_metrics.py --json results/Q3B/Q3B_vanilla_okvqa.json --metric gpt_judge --agg mean_non_zero
python run_metrics.py --json results/Q3B/Q3B_vanilla_textvqa.json --metric gpt_judge --agg mean_non_zero
python run_metrics.py --json results/Q3B/Q3B_vanilla_vizwiz.json --metric gpt_judge --agg mean_non_zero