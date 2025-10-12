import re
import string
import numpy as np

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
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
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