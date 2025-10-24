# Define custom prompt classes, and prompt answer extractors
import re

vanilla_prompt_suffix = (
            "Given the image, answer the question after <answer>."
        )

def vanilla_ans_extractor(raw_output):
    
    """Extract the final answer from GRIT's structured output"""
    # Look for answer after <answer> tag
    answer_match = re.search(r'<answer>\s*(.*?)(?:\s*$)', raw_output, re.DOTALL | re.IGNORECASE)

    if answer_match:
        return answer_match.group(1).strip()
    else:
        answer_match = re.search(r'answer\s*(.*?)(?:\s*$)', raw_output, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
    
    # If no structured output found, return the full output
    return raw_output.strip() 


grit_prompt_suffix = (
            " First, think between <think> and </think> while output necessary "
            "coordinates needed to answer the question in JSON with key 'bbox_2d'. "
            "Then, based on the thinking contents and coordinates, rethink between "
            "<rethink> </rethink> and then answer the question after <answer>."
        )

def grit_ans_extractor(raw_output):
    
    """Extract the final answer from GRIT's structured output"""
    # Look for answer after <answer> tag
    answer_match = re.search(r'<answer>\s*(.*?)(?:\s*$)', raw_output, re.DOTALL | re.IGNORECASE)
    # or look for word 'answer' in any case
    #answer_match = re.search(r'answer\s*(.*?)(?:\s*$)', grit_output, re.DOTALL | re.IGNORECASE)

    if answer_match:
        return answer_match.group(1).strip()
    else:
        answer_match = re.search(r'answer\s*(.*?)(?:\s*$)', raw_output, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
    
    # Fallback: look for content after </rethink>
    rethink_match = re.search(r'</rethink>\s*(.*?)(?:\s*$)', raw_output, re.DOTALL | re.IGNORECASE)
    if rethink_match:
        return rethink_match.group(1).strip()
    
    # If no structured output found, return the full output
    return raw_output.strip()

def parse_bounding_boxes(output: str):
    """Parse bounding boxes from GRIT output using regex"""
    bbox_regex = re.compile(r"\b\d+,\s*\d+,\s*\d+,\s*\d+\b")
    bboxes = []
    for match in bbox_regex.findall(output):
        try:
            x1, y1, x2, y2 = map(int, match.split(","))
            bboxes.append((x1, y1, x2, y2))
        except ValueError:
            pass
    return bboxes

sg_prompt_suffix = (
                    " Please generate a scene graph and Answer the Question based on the scene graph. "
                    "List all objects, their bounding boxes and how they relate to each other. "
                    "Generate a scene graph for this image in JSON format with the following structure:\n"
                    "{\n"
                    '    "objects": [\n'
                    "        {\n"
                    '            "id": "obj1",\n'
                    '            "name": "object_name",\n'
                    '            "bbox": "bounding box coordinates"\n'
                    "        }\n"
                    "    ],\n"
                    '    "relationships": [\n'
                    "        {\n"
                    '            "subject": "obj1",\n'
                    '            "predicate": "relationship_type",\n'
                    '            "object": "obj2"\n'
                    "        }\n"
                    "    ]\n"
                    "}\n"
                    "\n"
                    "Then, based on the scene graph, answer the question in single word or phrase.\n"
                    "OUTPUT FORMAT:\n"
                    "Question: {question}\n"
                    "JSON Scene Graph:\n"
                    "<answer>:"
                )

def sg_ans_extractor(raw_output):
    
    """Extract the final answer from scene graph structured output"""
    # Debug: print what we're trying to parse
    print(f"DEBUG sg_ans_extractor input: '{raw_output}'")
    
    # Look for "answer": "value" pattern (with quotes)
    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', raw_output, re.IGNORECASE)
    if answer_match:
        result = answer_match.group(1).strip()
        print(f"DEBUG extracted answer from quotes: '{result}'")
        return result
    
    # Look for "answer": value pattern (without quotes)
    answer_match = re.search(r'"answer"\s*:\s*([^,\n}]+)', raw_output, re.IGNORECASE)
    if answer_match:
        result = answer_match.group(1).strip()
        print(f"DEBUG extracted answer without quotes: '{result}'")
        return result
    
    # Fallback: Look for answer after "Answer:" (with colon)
    answer_match = re.search(r'Answer:\s*(.+)', raw_output, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Fallback: try without colon
    answer_match = re.search(r'Answer\s+(.+)', raw_output, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # If no structured output found, return the full output
    return raw_output.strip() 


PROMPTS_EXTRACTS = {
    "vanilla": {
        "prompt_suffix": vanilla_prompt_suffix, 
        "extract_func": vanilla_ans_extractor,
        "custom_func": None
    },
    "grit": {
        "prompt_suffix": grit_prompt_suffix, 
        "extract_func": grit_ans_extractor,
        "custom_func": parse_bounding_boxes
    }, 
    "sg": {
        "prompt_suffix": sg_prompt_suffix, 
        "extract_func": sg_ans_extractor,  # Reuse vanilla extractor for scene graph answers
        "custom_func": parse_bounding_boxes
    }
}

def create_prompt(task: str, question: str, post_prompt: str = "") -> str:
    """Create prompt based on task type"""
    if task not in PROMPTS_EXTRACTS:
        raise ValueError(f"Unsupported task: {task}")
    prompt_suffix = PROMPTS_EXTRACTS[task]["prompt_suffix"]
    if task == "vanilla":
        return f"Question: {question}{prompt_suffix}{post_prompt}\n"
    elif task == "grit":
        return f"Question: {question}{prompt_suffix}{post_prompt}\n"
    elif task == "sg": # POST PROMPT IGNORED FOR SG
<<<<<<< HEAD
        #return "Question: "+prompt_suffix.format(question=question)
        #import pdb; pdb.set_trace()
        return f"Question: {question}" + prompt_suffix
=======
        return "Question: "+prompt_suffix.replace("<question>", question)
>>>>>>> origin/main



