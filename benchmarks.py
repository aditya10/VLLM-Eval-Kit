from typing import Dict, Any, List

class BenchmarkConfig:
    """Configuration class for different benchmarks"""
    
    BENCHMARK_CONFIGS = {
        'vqav2': {
            'dataset_name': 'lmms-lab/VQAv2',
            'split': 'validation',
            'question_key': 'question',
            'image_key': 'image',
            'answer_key': 'multiple_choice_answer',
            'post_prompt': '\nAnswer the question using a single word or phrase.',
            'metric': 'exact_match',
            'description': 'VQA v2.0 dataset'
        },
        'gqa': {
            'dataset_name': 'lmms-lab/GQA',
            'dataset_config': 'testdev_balanced_instructions',
            'images_config': 'testdev_balanced_images',  # Separate config for images
            'split': 'testdev',  # Split name within the config
            'question_key': 'question',
            'image_key': 'image',  # Will be merged from images dataset
            'image_id_key': 'imageId',  # Key to match with images
            'answer_key': 'answer',
            'post_prompt': '\nAnswer the question using a single word or phrase.',
            'metric': 'exact_match',
            'description': 'GQA dataset',
            'requires_merge': True  # Flag to indicate this dataset needs special handling
        },
        'vizwiz': {
            'dataset_name': 'lmms-lab/VizWiz-VQA',
            'split': 'val',
            'question_key': 'question',
            'image_key': 'image',
            'answer_key': 'answers',
            'post_prompt': '\nWhen the provided information is insufficient, respond with \'Unanswerable\'.\nAnswer the question using a single word or phrase.',
            'metric': 'exact_match',
            'description': 'VizWiz-VQA dataset'
        },
        'textvqa': {
            'dataset_name': 'lmms-lab/TextVQA',
            'split': 'validation',
            'question_key': 'question',
            'image_key': 'image',
            'answer_key': 'answers',
            'post_prompt': '\nAnswer the question using a single word or phrase.',
            'metric': 'exact_match',
            'description': 'TextVQA dataset'
        },
        'infovqa': {
            'dataset_name': 'lmms-lab/DocVQA',
            'dataset_config': 'InfographicVQA',
            'split': 'validation',
            'question_key': 'question',
            'image_key': 'image',
            'answer_key': 'answers',
            'post_prompt': '\nAnswer the question using a single word or phrase.',
            'metric': 'anls',
            'description': 'InfoVQA dataset'
        },
        'pope': {
            'dataset_name': 'lmms-lab/POPE',
            'split': 'test',
            'question_key': 'question',
            'image_key': 'image',
            'answer_key': 'answer',
            'post_prompt': '\nAnswer the question using a single word or phrase.',
            'metric': 'exact_match', #TODO:Fix metric
            'description': 'POPE (Polling-based Object Probing Evaluation) dataset'
        },
        'okvqa': {
            'dataset_name': 'lmms-lab/OK-VQA',
            'split': 'val2014',
            'question_key': 'question',
            'image_key': 'image',
            'answer_key': 'answers',
            'post_prompt': '\nAnswer the question using a single word or phrase.',
            'metric': 'exact_match',
            'description': 'OK-VQA val2014 dataset'
        },
        'grit_vsr': {
            'dataset_name': '/home/mila/c/chandhos/scratch/data_cot/GRIT_data/vsr_val.jsonl',
            'split': 'val',
            'question_key': 'question',
            'image_key': 'image',
            'answer_key': 'answer',
            'img_folder': '/home/mila/c/chandhos/scratch/data_cot/GRIT_data/vsr',
            'post_prompt': '\nAnswer the question using a single word or phrase.',
            'metric': 'exact_match',
            'description': 'VSR Val dataset'
        },
        'grit_tallyqa': {
            'dataset_name': '/home/mila/c/chandhos/scratch/data_cot/GRIT_data/tallyqa_val.jsonl',
            'split': 'val',
            'question_key': 'question',
            'image_key': 'image',
            'answer_key': 'answer',
            'img_folder': '/home/mila/c/chandhos/scratch/data_cot/GRIT_data/tallyqa',
            'post_prompt': '\nAnswer the question using a single word or phrase.',
            'metric': 'exact_match',
            'description': 'TallyQA Val dataset'
        },
        'grit_mme': {
            'dataset_name': './datasets/mme_val.jsonl',
            'split': 'val',
            'question_key': 'question',
            'image_key': 'image',
            'answer_key': 'answer',
            'img_folder': '/home/aditya10/scratch/mme_images/mme',
            'post_prompt': '\nAnswer the question using a single word or phrase.',
            'metric': 'exact_match',
            'description': 'MME Val dataset from GRIT'
        },
        'grit_mathvista': {
            'dataset_name': './datasets/mathvista_mini_val.jsonl',
            'split': 'val',
            'question_key': 'question',
            'image_key': 'image',
            'answer_key': 'answer',
            'img_folder': '/home/aditya10/scratch/mathvista_mini_images/mathvista_mini',
            'post_prompt': '\nAnswer the question using a single word or phrase.',
            'metric': 'exact_match',
            'description': 'Mathvista Val dataset from GRIT'
        },
        'grit_gqa': {
            'dataset_name': './datasets/gqa_val.jsonl',
            'split': 'val',
            'question_key': 'question',
            'image_key': 'image',
            'answer_key': 'answer',
            'img_folder': '', #TODO: Add GQA images path
            'post_prompt': '\nAnswer the question using a single word or phrase.',
            'metric': 'exact_match',
            'description': 'GQA Val dataset from GRIT'
        },
        'grit_ovd_position': {
            'dataset_name': './datasets/ovd_position_val.jsonl',
            'split': 'val',
            'question_key': 'question',
            'image_key': 'image',
            'answer_key': 'answer',
            'img_folder': '/home/aditya10/scratch/ovd_position_images',
            'post_prompt': '\nAnswer the question using a single word or phrase.',
            'metric': 'exact_match',
            'description': 'GQA Val dataset from GRIT'
        },
    }
    
    @classmethod
    def get_config(cls, benchmark: str) -> Dict[str, Any]:
        """Get configuration for a specific benchmark"""
        if benchmark.lower() not in cls.BENCHMARK_CONFIGS:
            raise ValueError(f"Unsupported benchmark: {benchmark}. Supported: {list(cls.BENCHMARK_CONFIGS.keys())}")
        return cls.BENCHMARK_CONFIGS[benchmark.lower()]
    
    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all supported benchmarks"""
        return list(cls.BENCHMARK_CONFIGS.keys())