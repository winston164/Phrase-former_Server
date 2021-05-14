from transformers import (
    BartTokenizer, 
    BartForConditionalGeneration, 
    BartConfig, 
    BeamSearchScorer, 
    HammingDiversityLogitsProcessor, 
    MinLengthLogitsProcessor, 
    LogitsProcessorList,
    TemperatureLogitsWarper,
    LogitsProcessor,
    NoRepeatNGramLogitsProcessor
)
import language_tool_python 
from typing import List
import torch
import numpy as np
import traceback
from custom_logits_processors import DynamicBlockingProcessor

model = BartForConditionalGeneration.from_pretrained(
    'facebook/bart-large-cnn'
).cuda()

tokenizer = BartTokenizer.from_pretrained(
    'facebook/bart-large-cnn'
)

#tool = language_tool_python.LanguageTool('en-US')

num_beams = 50
num_beam_groups = 10

def getInputArgs(input_ids, diversity = 0.1):
    print(input_ids.shape)
    return {
        "input_ids": input_ids, 
        "min_length": input_ids.shape[-1] - 10 if input_ids.shape[-1] - 10 > 0 else 5,
        "num_beams": num_beams, 
        "no_repeat_ngram_size": 3,
        #"encoder_no_repeat_ngram_size":5,
        #"decoder_start_token_id": 0, 
        "num_return_sequences": num_beams,
        "num_beam_groups": num_beam_groups,
        "diversity_penalty": diversity,
        "bad_words_ids":[[tokenizer.convert_tokens_to_ids("CNN")] for _ in input_ids],
        "use_cache": True
    }

original_group_beam_search = BartForConditionalGeneration.group_beam_search


def getParaphrases(sentence: str, diversity: float = 0.1) -> List[str]:
    sentence = "" + sentence + ""
    input_ids = tokenizer([sentence], padding = True, max_length=1024, return_tensors='pt')[
    'input_ids'].cuda()

    generation_arguments = getInputArgs(input_ids, diversity)

    my_input_ids = input_ids
    def g(self, input_ids, beam_scorer, **kwargs):
        my_input_ids
        kwargs["logits_processor"].insert(0, DynamicBlockingProcessor(my_input_ids, num_beams, num_beam_groups, 0.5))
        #kwargs["logits_processor"].insert(1, NoRepeatNGramLogitsProcessor(2))
        return original_group_beam_search(self, input_ids, beam_scorer, **kwargs)

    BartForConditionalGeneration.group_beam_search = g

    outputs = model.generate(
        **generation_arguments
    )

    torch.cuda.empty_cache()

    return_sentences =  tokenizer.batch_decode(outputs, skip_special_tokens = True)
    return return_sentences #[language_tool_python.utils.correct(s, tool.check(s)) for s in return_sentences]

s = "In the town where I was born, lived a man who sailed to sea. And he told us all his life in a land of submarine."