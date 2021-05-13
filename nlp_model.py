from transformers import (
    MBartTokenizer, 
    MBartForConditionalGeneration, 
    MBartConfig, 
    BeamSearchScorer, 
    HammingDiversityLogitsProcessor, 
    MinLengthLogitsProcessor, 
    LogitsProcessorList,
    TemperatureLogitsWarper,
    LogitsProcessor,
    NoRepeatNGramLogitsProcessor
)
from typing import List
import torch
import numpy as np
import traceback
from custom_logits_processors import DynamicBlockingProcessor

model = MBartForConditionalGeneration.from_pretrained(
    'facebook/mbart-large-cc25'
).cuda()

tokenizer = MBartTokenizer.from_pretrained(
    'facebook/mbart-large-cc25'
)

# example = ''' </s> The seven steps to a coding career. </s> '''

# input_ids = tokenizer([example], padding = True, max_length=1024, return_tensors='pt')[
#     'input_ids'].cuda()


# Group Beam Search test
num_beams = 50
num_beam_groups = 10

def getInputArgs(input_ids):
    return {
        "input_ids": input_ids, 
        "min_length": input_ids.shape[-1] - 10 if input_ids.shape[-1] - 10 > 0 else 5,
        "num_beams": num_beams, 
        "no_repeat_ngram_size": 3,
        "encoder_no_repeat_ngram_size":5,
        "decoder_start_token_id": 1, 
        "num_return_sequences": num_beams,
        "num_beam_groups": num_beam_groups,
        "diversity_penalty": 0.1,
        "use_cache": True
    }

original_group_beam_search = MBartForConditionalGeneration.group_beam_search


# my_input_ids = input_ids
# def g(self, input_ids, beam_scorer, **kwargs):
#     global my_input_ids
#     kwargs["logits_processor"].insert(0, DynamicBlockingProcessor(my_input_ids, num_beams, num_beam_groups, 0.5))
#     kwargs["logits_processor"].insert(1, NoRepeatNGramLogitsProcessor(2))
#     return original_group_beam_search(self, input_ids, beam_scorer, **kwargs)

# MBartForConditionalGeneration.group_beam_search = g

# outputs = model.generate(
#     **generate_args
#     )


# for i, sentence in enumerate(tokenizer.batch_decode(outputs, 
# skip_special_tokens=True
# )):
    
#     if (i % (num_beams//num_beam_groups)) == 0 :print(sentence)


def getParaphrases(sentence: str) -> List[str]:
    sentence = " </s> " + sentence + " </s> "
    input_ids = tokenizer([sentence], padding = True, max_length=1024, return_tensors='pt')[
    'input_ids'].cuda()

    generation_arguments = getInputArgs(input_ids)

    my_input_ids = input_ids
    def g(self, input_ids, beam_scorer, **kwargs):
        my_input_ids
        kwargs["logits_processor"].insert(0, DynamicBlockingProcessor(my_input_ids, num_beams, num_beam_groups, 0.5))
        kwargs["logits_processor"].insert(1, NoRepeatNGramLogitsProcessor(2))
        return original_group_beam_search(self, input_ids, beam_scorer, **kwargs)

    MBartForConditionalGeneration.group_beam_search = g

    outputs = model.generate(
        **generation_arguments
    )

    torch.cuda.empty_cache()

    return tokenizer.batch_decode(outputs, skip_special_tokens = True)


    



# oldFunc = MBartForConditionalGeneration._update_model_kwargs_for_generation


# @staticmethod
# def newFunc(outputs, model_kwargs, is_encoder_decoder = False):
# #    global keep_objects
# #    current_objects = []
# #    for obj in gc.get_objects():
# #        try:
# #            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
# #                if (obj not in keep_objects):
# #                    current_objects.append(obj)
# #                    print(type(obj), obj.size())
# #        
# #        except:
# #            pass
        
#     input("press any")
#     model_kwargs["use_cache"] = True
#     torch.cuda.empty_cache()
#     return oldFunc(outputs, model_kwargs, is_encoder_decoder)

# MBartForConditionalGeneration._update_model_kwargs_for_generation = newFunc

# Manual group_beam_search call
# model_kwargs = {
#     "output_attentions": False,
#     "output_hidden_states": False,
#     "attention_mask": model._prepare_attention_mask_for_generation(
#         input_ids, model.config.pad_token_id, model.config.bos_token_id
#     ),
# }

# model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

# model_kwargs["use_cache"] = True

# decoder_input_ids = model._prepare_decoder_input_ids_for_generation(
#     input_ids, model.config.decoder_start_token_id, model.config.bos_token_id
# )

# logits_processor = model._get_logits_processor(
#     repetition_penalty = None,
#     no_repeat_ngram_size=3,
#     encoder_no_repeat_ngram_size = None,
#     encoder_input_ids = input_ids,
#     bad_words_ids = None,
#     min_length = input_ids.shape[-1],
#     max_length = 1024,
#     eos_token_id = model.config.eos_token_id,
#     forced_bos_token_id=None,
#     forced_eos_token_id=None,
#     prefix_allowed_tokens_fn= None,
#     num_beams = num_beams,
#     num_beam_groups = num_beam_groups,
#     diversity_penalty= 0.9,
#     remove_invalid_values= None
# )

# stopping_criteria = model._get_stopping_criteria(
#     max_length = 1024,
#     max_time = None
# )


# batch_size = decoder_input_ids.shape[0]
# early_stopping = True

# diverse_beam_scorer = BeamSearchScorer(
#     batch_size = batch_size,
#     max_length = 1024,
#     num_beams = num_beams,
#     device = model.device,
#     num_beam_hyps_to_keep = num_beams,
#     num_beam_groups = num_beam_groups
# )

# decoder_input_ids, model_kwargs = model._expand_inputs_for_generation(
#     decoder_input_ids, 
#     expand_size=num_beams, 
#     is_encoder_decoder = model.config.is_encoder_decoder, 
#     **model_kwargs
# )

# group_beam_search_args = {
#     "input_ids": decoder_input_ids,
#     "beam_scorer": diverse_beam_scorer,
#     "logits_processor": logits_processor,
#     "stopping_criteria": stopping_criteria,
#     "max_length": 1024,
#     "pad_token_id": model.config.pad_token_id,
#     "eos_token_id": model.config.eos_token_id,
#     "output_scores": False,
#     "return_dict_in_generate": model.config.return_dict_in_generate,
#     **model_kwargs
# }

# try:
#     out = model.group_beam_search(**group_beam_search_args)

# except :
#     traceback.print_exc()


# for sentence in tokenizer.batch_decode(out, skip_special_tokens=True):
#     print(sentence)