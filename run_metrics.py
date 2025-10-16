import json
from metrics import exact_match_hf_evaluate, anls, gpt_judge_metric
from benchmarks import BenchmarkConfig
import argparse


def run_metric(json_file, metric): 

    with open(json_file, 'r') as f:
        results = json.load(f)

    scores = []
    for entry in results['detailed_results']:
        predicted = entry['predicted_answer']
        ground_truths = entry['ground_truth_answers']
        question = entry['question']

        if metric == 'exact_match':
            score = exact_match_hf_evaluate([predicted], ground_truths)['exact_match']
        elif metric == 'anls':
            score = anls(ground_truths, [predicted])['anls']
        elif metric == 'gpt_judge':
            score = gpt_judge_metric(question, predicted, ground_truths)['gpt_judge_score']

        entry['final_score'] = score

        scores.append(score)
    
    overall_score = sum(scores) / len(scores) if scores else 0.0
    results['accuracy'] = overall_score

    # Save
    with open('final/'+json_file.split('/')[-1], 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run metrics on VQA results")
    parser.add_argument('--json', type=str, help='Path to the JSON file with results', default='results/Grit_grit_okvqa.json')
    parser.add_argument('--metric', type=str, choices=['exact_match', 'anls', 'gpt_judge'], help='Metric to compute', default='exact_match')
    args = parser.parse_args()

    run_metric(args.json, args.metric)

   