#!/usr/bin/env python3
"""
VLM Multi-Benchmark Evaluation Suite
This script evaluates the Qwen and Qwen-like models (e.g. GRIT) on multiple VQA benchmarks using lmms-eval compatible interface.
Supports: VQAv2, GQA, VizWiz, TextVQA, InfoVQA, POPE, OKVQA
"""

import re
import torch
import json
import os
from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
#from transformers.modeling_utils import VLMS
from tqdm import tqdm
import argparse
from datetime import datetime
import numpy as np
import random
import logging

from benchmarks import BenchmarkConfig
from custom_prompts import PROMPTS_EXTRACTS, create_prompt
from vision_process import process_vision_info
from metrics import exact_match_hf_evaluate, gpt_judge_metric

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VLLMEvaluator:
    """Main evaluator class with support for multiple benchmarks"""
    
    def __init__(self, 
                 model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", 
                 device: str = "auto", 
                 batch_size: int = 16,
                 max_new_tokens: int = 1024,
                 task: str = "vanilla"):
        """Initialize the GRIT model and processor
        
        Args:
            model_id: Hugging Face model ID
            device: Device to use (auto, cpu, cuda)
            batch_size: Batch size for processing
            max_new_tokens: Maximum number of tokens to generate
            task: Task type (grit, vanilla, sg)
        """
        logger.info(f"Loading model: {model_id}")
        
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.task = task
        
        # Load model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map={"": 0} if device != "cpu" else "cpu",
            attn_implementation="flash_attention_2",
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Set padding side to left for Flash Attention compatibility
        self.processor.tokenizer.padding_side = 'left'
        self.model.generation_config.use_cache = True
        #self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
        
        # Configure generation settings
        self.model.generation_config.max_new_tokens = max_new_tokens

        if task == "grit":
            self.model.generation_config.temperature = 0.001
            self.model.generation_config.top_k = 1
            self.model.generation_config.top_p = 0.0
        if task == "sg":
            self.model.generation_config.max_new_tokens = 380
            self.model.generation_config.temperature = 0.001
            self.model.generation_config.top_k = 1
            self.model.generation_config.top_p = 0.0
            #self.model.generation_config.use_cache = False
        if task == "vanilla":
            self.model.generation_config.max_new_tokens = 380
            self.model.generation_config.temperature = 0.001
            self.model.generation_config.top_k = 1
            self.model.generation_config.top_p = 0.0
            #self.model.generation_config.use_cache = False

        # Load custom prompts and extractors
        if task not in PROMPTS_EXTRACTS:
            raise ValueError(f"Unsupported task: {task}")
        
        self.extract_final_answer = PROMPTS_EXTRACTS[task]["extract_func"]
        self.custom_func = PROMPTS_EXTRACTS[task]["custom_func"] if "custom_func" in PROMPTS_EXTRACTS[task] else None
        
        logger.info(f"Model loaded successfully! Batch size: {batch_size}")
        logger.info(f"Task type: {task}")


    def predict_batch(self, batch_data: List[Dict], post_prompt: str = "") -> List[Dict[str, Any]]:
        """Make predictions on a batch of samples for better GPU utilization"""
        if not batch_data:
            return []
        
        
        texts = []
        images = []
        valid_indices = []  # Track which original items are valid
        
        # Prepare batch
        for i, item in enumerate(batch_data):
            image = item["image"]
            question = item["question"]
            
            if image is None or not question:
                continue
            #import pdb; pdb.set_trace()
            prompt_text = create_prompt(self.task, question, post_prompt)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            texts.append(chat_text)
            images.append(image)
            valid_indices.append(i)
        
        if not texts:
            # Return error results for all items if none are valid
            return [{"raw_output": "", "final_answer": "", "bboxes": [], "success": False, "error": "Invalid image or question"} for _ in batch_data]
        
        # Process all images efficiently for batch
        batch_messages = []
        for i, (image, text) in enumerate(zip(images, texts)):
            batch_messages.append({
                "role": "user", 
                "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]
            })
        
        # Process vision info for all images at once
        all_img_inputs = []
        for msg in batch_messages:
            img_inputs, _ = process_vision_info([msg])
            all_img_inputs.extend(img_inputs if img_inputs else [])
        
        # Batch processing
        inputs = self.processor(
            text=texts,
            images=all_img_inputs if all_img_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        from time import time
        start_time = time()
        # Generate responses in batch
        with torch.inference_mode():
            gen_ids = self.model.generate(**inputs, generation_config=self.model.generation_config)
        
        # Decode all outputs
        raw_outputs = self.processor.batch_decode(
            gen_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(raw_outputs)
        end_time = time()
        print(f"Time taken: {end_time - start_time} seconds")
        #import pdb; pdb.set_trace()
        
        # Process results - need to align with original batch_data indices
        batch_results = []
        prediction_idx = 0
        
        for i in range(len(batch_data)):
            if i in valid_indices:
                # This item was processed successfully
                raw_output = raw_outputs[prediction_idx]
                final_answer = self.extract_final_answer(raw_output)
                if self.custom_func:
                    bboxes = self.custom_func(raw_output)
                else:
                    bboxes = []
                batch_results.append({
                    "raw_output": raw_output,
                    "final_answer": final_answer,
                    "bboxes": bboxes,
                    "success": True,
                    "error": None
                })
                prediction_idx += 1
            else:
                # This item was skipped due to invalid image/question
                batch_results.append({
                    "raw_output": "",
                    "final_answer": "",
                    "bboxes": [],
                    "success": False,
                    "error": "Invalid image or question"
                })
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return batch_results
        
        # except Exception as e:
        #     logger.error(f"Error in batch prediction: {e}")
        #     # Clear GPU memory on error
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        #     # Return error for all items in batch
        #     return [{"raw_output": "", "final_answer": "", "bboxes": [], "success": False, "error": str(e)} for _ in batch_data]

    def evaluate_benchmark(self, 
                          benchmark: str,
                          limit: int = None,
                          output_dir: str = "./results",
                          random_seed: int = 42) -> Dict[str, Any]:
        """Evaluate on a specific benchmark"""
        
        # Get benchmark configuration
        config = BenchmarkConfig.get_config(benchmark)
        logger.info(f"Evaluating on {config['description']}")
        if config.get('requires_merge', False):
            logger.info(f"Dataset: {config['dataset_name']}, Merging configs: {config['dataset_config']} + {config['images_config']}")
        elif 'dataset_config' in config and config['dataset_config']:
            logger.info(f"Dataset: {config['dataset_name']}, Config: {config['dataset_config']}")
        else:
            logger.info(f"Dataset: {config['dataset_name']}, Split: {config['split']}")
        
        # Load dataset
        try:
            if config.get('requires_merge', False):
                # Special handling for GQA - merge instructions and images
                instructions_dataset = load_dataset(
                    config['dataset_name'], 
                    config['dataset_config'], 
                    split=config['split']
                )
                images_dataset = load_dataset(
                    config['dataset_name'], 
                    config['images_config'], 
                    split=config['split']
                )
                
                # Create a mapping from image id to image
                image_map = {item['id']: item['image'] for item in images_dataset}
                
                # Merge datasets by adding images to instructions
                merged_data = []
                for item in instructions_dataset:
                    image_id = item[config['image_id_key']]
                    if image_id in image_map:
                        merged_item = dict(item)
                        merged_item[config['image_key']] = image_map[image_id]
                        merged_data.append(merged_item)
                
                # Convert back to dataset format
                from datasets import Dataset
                dataset = Dataset.from_list(merged_data)
                
            elif 'dataset_config' in config and config['dataset_config']:
                # Load with specific configuration
                if config['split']:
                    dataset = load_dataset(config['dataset_name'], config['dataset_config'], split=config['split'])
                else:
                    # For datasets where split is included in config name
                    dataset = load_dataset(config['dataset_name'], config['dataset_config'])
            else:
                # Load without configuration (original behavior)
                dataset = load_dataset(config['dataset_name'], split=config['split'])
            if limit:
                # Set random seed for consistent sampling
                np.random.seed(random_seed)
                random.seed(random_seed)
                
                total_samples = len(dataset)
                if limit >= total_samples:
                    logger.info(f"Limit ({limit}) >= dataset size ({total_samples}), using all samples")
                else:
                    # Generate random indices without replacement
                    random_indices = np.random.choice(total_samples, size=limit, replace=False)
                    random_indices = sorted(random_indices.tolist())
                    dataset = dataset.select(random_indices)
                    logger.info(f"Randomly sampled {limit} samples from {total_samples} total samples (seed: {random_seed})")
            
            logger.info(f"Dataset loaded: {len(dataset)} samples")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return {"error": str(e)}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare batches
        batches = []
        batch_data = []
        
        for i, sample in enumerate(dataset):
            batch_item = {
                "sample_id": i,
                "image": sample.get(config['image_key'], None),
                "question": sample.get(config['question_key'], ""),
                "answers": sample.get(config['answer_key'], [])
            }
            #If GT is a string, convert to list
            if not isinstance(batch_item["answers"], list):
                batch_item["answers"] = [batch_item["answers"]]
            batch_data.append(batch_item)
            
            # Create batch when batch_size is reached
            if len(batch_data) == self.batch_size:
                batches.append(batch_data)
                batch_data = []
        
        # Add remaining samples
        if batch_data:
            batches.append(batch_data)
        
        logger.info(f"Processing {len(batches)} batches of size {self.batch_size}")
        
        # Evaluation results
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        # Process batches
        for batch_idx, batch in enumerate(tqdm(batches, desc=f"Processing {benchmark} batches")):
            # Get predictions for the batch
            predictions = self.predict_batch(batch, config['post_prompt'])
            
            # Process results
            for item, prediction in zip(batch, predictions):
                if not prediction["success"]:
                    continue
                
                predicted_answer = prediction["final_answer"]
                ground_truth_answers = item["answers"]
                
                metric_list = config['metric'].split('+')
                match_score = {}
                # Calculate accuracy based on metric type
                if 'exact_match' in metric_list:
                    match_score.update(exact_match_hf_evaluate([predicted_answer]*len(ground_truth_answers), ground_truth_answers, ignore_case=True, ignore_punctuation=True))
                
                if 'gpt_judge' in metric_list:
                    match_score.update(gpt_judge_metric(item["question"], predicted_answer, ground_truth_answers, model='gpt-4o-mini'))

                # if 'llama_judge' in metric_list:
                #     match_score.update(llama_judge_metric(item["question"], predicted_answer, ground_truth_answers))
                
                is_correct = False
                if any([s > 0 for s in match_score.values()]):
                    is_correct = True
                    correct_predictions += 1
                total_predictions += 1
                
                # Store result
                result = {
                    "sample_id": item["sample_id"],
                    "question": item["question"],
                    "ground_truth_answers": item["answers"],
                    "predicted_answer": prediction["final_answer"],
                    "raw_output": prediction["raw_output"],
                    "bboxes": prediction.get("bboxes", []),
                    "is_correct": is_correct,
                    "match_score": match_score,
                    "success": prediction["success"],
                    "error": prediction["error"]
                }
                results.append(result)
            
            # Print progress
            if (batch_idx + 1) % 1 == 0:
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                logger.info(f"Progress: {batch_idx+1}/{len(batches)} batches, Accuracy so far: {accuracy:.4f}")
        
        # Calculate final metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        evaluation_results = {
            "dataset": config['dataset_name'],
            "dataset_config": config.get('dataset_config', None),
            "split": config['split'],
            "total_samples": len(dataset),
            "successful_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "metric": config['metric'],
            "batch_size": self.batch_size,
            "limit": limit,
            "random_seed": random_seed if limit else None,
            "timestamp": datetime.now().isoformat(),
            "model_id": self.model_id,
            "task": self.task,
            "detailed_results": results
        }
        
        # Save results
        mname=self.model_id.split('/')[-1]
        output_file = os.path.join(output_dir, f"{mname}_{self.task}_{benchmark}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"\nEvaluation completed for {benchmark}!")
        logger.info(f"Dataset: {config['dataset_name']}")
        if 'dataset_config' in config and config['dataset_config']:
            logger.info(f"Config: {config['dataset_config']}")
        else:
            logger.info(f"Split: {config['split']}")
        logger.info(f"Total samples: {len(dataset)}")
        logger.info(f"Successful predictions: {total_predictions}")
        logger.info(f"Correct predictions: {correct_predictions}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Metric: {config['metric']}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Results saved to: {output_file}")
        
        return evaluation_results
    
    def evaluate_all_benchmarks(self,
                               benchmarks: List[str] = None,
                               limit: int = None,
                               output_dir: str = "./results",
                               random_seed: int = 42) -> Dict[str, Any]:
        """Evaluate on all specified benchmarks"""
        
        if benchmarks is None:
            benchmarks = BenchmarkConfig.list_benchmarks()
        
        logger.info(f"Starting evaluation on benchmarks: {benchmarks}")
        
        all_results = {}
        summary = {
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat(),
            "benchmarks_evaluated": benchmarks,
            "limit": limit,
            "random_seed": random_seed if limit else None,
            "results": {}
        }
        
        for benchmark in benchmarks:
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting evaluation on {benchmark.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                result = self.evaluate_benchmark(
                    benchmark=benchmark,
                    limit=limit,
                    output_dir=output_dir,
                    random_seed=random_seed
                )
                
                all_results[benchmark] = result
                summary["results"][benchmark] = {
                    "dataset": result["dataset"],
                    "dataset_config": result.get("dataset_config", None),
                    "split": result["split"],
                    "accuracy": result["accuracy"],
                    "total_samples": result["total_samples"],
                    "successful_predictions": result["successful_predictions"],
                    "correct_predictions": result["correct_predictions"],
                    "metric": result["metric"],
                    "batch_size": result["batch_size"],
                    "timestamp": result["timestamp"]
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {benchmark}: {e}")
                all_results[benchmark] = {"error": str(e)}
                summary["results"][benchmark] = {"error": str(e)}
        
        # Save summary
        summary_file = os.path.join(output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*50}")
        logger.info("EVALUATION SUMMARY")
        logger.info(f"{'='*50}")
        for benchmark, result in summary["results"].items():
            if "error" not in result:
                logger.info(f"{benchmark.upper()}: {result['accuracy']:.4f} ({result['correct_predictions']}/{result['successful_predictions']}/{result['total_samples']}) - {result['dataset']}")
            else:
                logger.info(f"{benchmark.upper()}: ERROR - {result['error']}")
        
        logger.info(f"\nSummary saved to: {summary_file}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate GRIT model on multiple VQA benchmarks")
    parser.add_argument("--model_id", default="yfan1997/GRIT-20-Qwen2.5-VL-3B", 
                       help="Hugging Face model ID")
    parser.add_argument("--benchmarks", nargs='+', 
                       choices=BenchmarkConfig.list_benchmarks() + ['all'],
                       default=['all'],
                       help="Benchmarks to evaluate on")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit number of samples for testing")
    parser.add_argument("--output_dir", default="./output/results", 
                       help="Output directory for results")
    parser.add_argument("--device", default="auto", 
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing (default: 4)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for consistent sampling when using --limit (default: 42)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                       help="Maximum number of new tokens to generate (default: 1024)")
    parser.add_argument("--task", type=str, default="grit",
                       choices=['grit', 'vanilla', 'sg'],
                       help="Task type: grit (thinking structure), vanilla (direct answer), sg (scene graph) (default: grit)")
    
    args = parser.parse_args()
    
    # Handle 'all' benchmarks
    if 'all' in args.benchmarks:
        benchmarks = BenchmarkConfig.list_benchmarks()
    else:
        benchmarks = args.benchmarks
    
    logger.info(f"Initializing evaluator...")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Benchmarks: {benchmarks}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Task: {args.task}")
    
    # Initialize evaluator
    evaluator = VLLMEvaluator(
        model_id=args.model_id, 
        device=args.device, 
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        task=args.task
    )
    
    # Run evaluation
    results = evaluator.evaluate_all_benchmarks(
        benchmarks=benchmarks,
        limit=args.limit,
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )
    
    logger.info("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
