import re
import string
import numpy as np
from openai import OpenAI
from keys import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


# Stop words for word overlap matching
# Filters out common function words and VQA-specific filler words
STOP_WORDS = {
    # Articles
    'a', 'an', 'the',
    
    # Common prepositions
    'at', 'by', 'for', 'from', 'in', 'of', 'on', 'to', 'with', 'into', 'onto',
    'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
    'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'during', 'through', 'toward', 'under', 'until', 'upon', 'within', 'without',
    
    # Common conjunctions
    'and', 'or', 'but', 'nor', 'so', 'yet',
    
    # Common pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their',
    'mine', 'yours', 'hers', 'ours', 'theirs',
    'this', 'that', 'these', 'those',
    
    # Common verbs (be, have, do)
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'done',
    
    # Common auxiliary verbs
    'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
    
    # Question words
    'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
    
    # Other common words
    'if', 'as', 'than', 'then', 'there', 'here',
    'not', 'no', 'yes',  # Note: keeping these short but might be meaningful in some contexts
    
    # VQA-specific filler words
    'some', 'sort', 'kind', 'type', 'thing',  # "some sort error" → focus on "error"
    'picture', 'image', 'photo', 'photograph',  # "picture on computer" → focus on "computer"
    'unanswerable',  # VizWiz specific - not a real answer
    'cannot', 'cant', 'can not',  # Negation fillers
}

### the code used in the `exact_match_hf_evaluate` function is ported from
### https://github.com/huggingface/evaluate/blob/main/metrics/exact_match/exact_match.py
### which is under the apache license.

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def exact_match_hf_evaluate(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=True,
    ignore_punctuation=True,
    ignore_numbers=True,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}

def parse_llm_match_score(response_text):

    import json
    try:
        score_data = json.loads(response_text)
        score = float(score_data.get('score', 0.0))
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract score with regex
        import re
        score_match = re.search(r'"score"?\s*:\s*([0-9.]+)', response_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            score = 0.0
    
    # Ensure score is within valid range [0, 1]
    score = max(0.0, min(1.0, score))
    return score



def gpt_judge_metric(question, prediction, references, model='gpt-4o-mini'):

    prompt = f"""You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer set, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the json form "{{"score": <score>}}". The evaluation criteria require that the closer the model's answer is to any of the standard answers, the higher the score.
            Question: {question} 
            Standard answer: {references} 
            Model's answer: {prediction}"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,  # For consistent scoring
            max_tokens=100    # Short response expected
        )
        
        # Extract the response content
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON response
        score = parse_llm_match_score(response_text)
        
        
        return {"gpt_judge_score": score}
        
    except Exception as e:
        print(f"Error in GPT judge evaluation: {e}")
        return {"gpt_judge_score": 0.0}
    

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def anls(
    references,
    predictions,
    thresh_hold=0.5,
):
    """https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py"""
    values = []
    # Unwrap predictions if it's a nested list
    pred = predictions[0] if isinstance(predictions[0], str) else predictions[0][0]

    for answer in references:
        # preprocess both the answers - gt and prediction
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred.strip().lower().split())

        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    question_result = 1 - min(values)

    if question_result < thresh_hold:
        question_result = 0
    return {"anls": question_result}