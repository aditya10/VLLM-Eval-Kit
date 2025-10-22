from metrics import gpt_judge_batch
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT Judge Metric Batch Job")
    parser.add_argument('--json', type=str, help='Path to the input JSON file', required=True)
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        input_data = json.load(f)

    input_data_entries = input_data['detailed_results']

    gpt_judge_batch(input_data_entries)