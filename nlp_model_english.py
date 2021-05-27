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
from typing import List, Tuple
import torch
import numpy as np
import traceback
from custom_logits_processors import DynamicBlockingProcessor
from bert_score import BERTScorer

model = BartForConditionalGeneration.from_pretrained(
    'facebook/bart-large-cnn'
).cuda()

tokenizer = BartTokenizer.from_pretrained(
    'facebook/bart-large-cnn'
)

scorer = BERTScorer(lang="en", rescale_with_baseline=True)

#tool = language_tool_python.LanguageTool('en-US')

num_beams = 50
num_beam_groups = 10

def getBannedWords(s : str, banned_strings: List[str] = []) ->  List[List[int]] :
    # Here we represent a word as a list of ids representing the string 
    # that makes the word. 

    # Ban capitalizations
    # if "money" is in the string s, ban "Money"
    words_strings = s.split(" ")
    capitalized_words_strings = [word_string.capitalize() for word_string in words_strings]
    excluded_capitalized_words_srings = list(set(capitalized_words_strings).difference(set(words_strings)))
    excluded_capitalized_words = [
        tokenizer(word_string, add_prefix_space = True, add_special_tokens = False).input_ids
        for word_string in excluded_capitalized_words_srings
        ]
    
    # Banned strings to words
    banned_words = [
        tokenizer(word_string, add_prefix_space = True, add_special_tokens = False).input_ids
        for word_string in banned_strings
        ]

    return excluded_capitalized_words + banned_words


def strings_as_suffix_words(strings: List[str]):
    return [
        tokenizer(string, add_prefix_space = False, add_special_tokens = False).input_ids
        for string in strings
    ]

def getInputArgs(input_ids, diversity = 0.1, bannedIds: List[List[int]] = []):
    if(len(bannedIds) == 0): bannedIds = [[16256] for _ in input_ids]
    return {
        "input_ids": input_ids, 
        "min_length": (input_ids.shape[-1] - 10 ) if ((input_ids.shape[-1] - 10) > 0) else 5,
        "num_beams": num_beams, 
        "no_repeat_ngram_size": 3,
        #"encoder_no_repeat_ngram_size":5,
        #"decoder_start_token_id": 0, 
        "num_return_sequences": num_beams,
        "num_beam_groups": num_beam_groups,
        "diversity_penalty": diversity,
        "bad_words_ids":bannedIds,
        "use_cache": True
    }

original_group_beam_search = BartForConditionalGeneration.group_beam_search

default_banned_strings = ["CNN", '"', "'"]

def getParaphrases(sentence: str, diversity: float = 0.1, banned_words: List[str] = []) -> List[str]:
    sentence = "" + sentence + ""
    input_ids = tokenizer([sentence], padding = True, max_length=1024, return_tensors='pt')[
    'input_ids'].cuda()

    default_banns = [string for string in default_banned_strings if (sentence.find(string) == -1)]
    
    capitalized_tokens = getBannedWords(sentence, default_banns + banned_words)

    banned_tokens = capitalized_tokens + strings_as_suffix_words(default_banns)

    generation_arguments = getInputArgs(input_ids, diversity, banned_tokens)

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

    return_sentences = list(set(return_sentences))

    sentence_to_index  =  {}
    for i, sen in enumerate(return_sentences):
        if sen in sentence_to_index :
            print(sen, i, sentence_to_index[sentence])
        sentence_to_index[sen] = i

    #P,R,F1 = scorer.score(return_sentences, [sentence] * len(return_sentences))

    #return_sentences.sort(key = lambda s: F1[sentence_to_index[s]], reverse = True)
    return_sentences.sort(key = lambda s: scorer.score([s],[sentence])[2][0], reverse = True)
    # for sen in return_sentences:
    #     index = sentence_to_index[sen]
    #     print(F1[index], P[index], R[index], sen)

    return return_sentences #[language_tool_python.utils.correct(s, tool.check(s)) for s in return_sentences]

s = "In the town where I was born, lived a man who sailed to sea. And he told us all his life in a land of submarine."