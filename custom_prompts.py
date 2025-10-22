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

sg_prompt_suffix = """Please generate a scene graph and Answer the Question based on the scene graph. 
    List the main objects, their bounding boxes and how they relate to each other. 
    "Generate a scene graph for this image in JSON format with the following structure:
    {
        "objects": [
            {
                "id": "obj1",
                "name": "object_name",
                "bbox": "bounding box coordinates"
            }
        ],
        "relationships": [
            {
                "subject": "obj1",
                "predicate": "relationship_type",
                "object": "obj2"
            }
        ],
    }
    
    Then, based on the scene graph, answer the question.\n
    OUTPUT FORMAT:
    Question: <question>
    JSON Scene Graph:
    Answer:
    """


def sg_ans_extractor(raw_output):
    
    """Extract the final answer from GRIT's structured output"""
    # Look for answer after <answer> tag
    answer_match = re.search(r'Answer\s*(.*?)(?:\s*$)', raw_output, re.DOTALL | re.IGNORECASE)

    if answer_match:
        return answer_match.group(1).strip()
    else:
        answer_match = re.search(r'Answer\s*(.*?)(?:\s*$)', raw_output, re.DOTALL | re.IGNORECASE)
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
        return "Question: "+prompt_suffix.replace("<question>", question)



