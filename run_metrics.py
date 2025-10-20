import json
from metrics import exact_match_hf_evaluate, anls, gpt_judge_metric
from benchmarks import BenchmarkConfig
import argparse
import time

def run_metric(json_file, metric, agg='mean'): 

    print(f"Running {metric} metric on {json_file} with aggregation {agg}")

    with open(json_file, 'r') as f:
        results = json.load(f)

    scores = []
    for entry in results['detailed_results']:
        predicted = entry['predicted_answer']
        ground_truths = entry['ground_truth_answers']
        question = entry['question']

        if metric == 'exact_match':
            score_d = exact_match_hf_evaluate([predicted]*len(ground_truths), ground_truths)
            score = score_d['exact_match']
        elif metric == 'anls':
            score_d = anls(ground_truths, [predicted])
            score = score_d['anls']
        elif metric == 'gpt_judge':
            if 'gpt_judge_score' in entry['match_score']:
                score_d = {'gpt_judge_score': entry['match_score']['gpt_judge_score']}
                score = score_d['gpt_judge_score']
            else:
                score_d = gpt_judge_metric(question, predicted, ground_truths)
                score = score_d['gpt_judge_score']

        entry['match_score'].update(score_d)
        scores.append(score)
    
    if agg == 'mean':
        overall_score = sum(scores) / len(scores) if scores else 0.0
    if agg == 'mean_non_zero':
        non_zero_scores = [s for s in scores if s > 0]
        overall_score = len(non_zero_scores) / len(scores)
    
    if "agg_results" in results:
        results['agg_results'][metric] = overall_score
    else:
        results['agg_results'] = {metric: overall_score}
    
    # Save
    with open('final/'+json_file.split('/')[-1], 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run metrics on VQA results")
    parser.add_argument('--json', type=str, help='Path to the JSON file with results', default='results/Q7B_vanilla_infovqa.json')
    parser.add_argument('--metric', type=str, choices=['exact_match', 'anls', 'gpt_judge'], help='Metric to compute', default='gpt_judge')
    parser.add_argument('--agg', type=str, choices=['mean', 'mean_non_zero'], help='Aggregation method', default='mean')
    args = parser.parse_args()

    run_metric(args.json, args.metric, args.agg)

   