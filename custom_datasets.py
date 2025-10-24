import json
import os
from PIL import Image
from datasets import Dataset

def load_grit_jsonl_dataset(config):
    """Load VSR dataset from local JSONL file"""
    dataset_path = config['dataset_name']
    img_folder = os.path.expanduser(config['img_folder'])
    
    # Load JSONL data
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    
    # Process the data to match expected format
    processed_data = []
    for item in data:
        # Load image from the specified folder
        image_path = os.path.join(img_folder, item[config['image_key']])
        
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                processed_item = {
                    config['question_key']: item[config['question_key']],
                    config['image_key']: image,
                    config['answer_key']: str(item[config['answer_key']])
                }
                processed_data.append(processed_item)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
        else:
            print(f"Image not found: {image_path}")
            continue
    
    dataset = Dataset.from_list(processed_data)
    return dataset
