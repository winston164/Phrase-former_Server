from transformers import (
    MBartTokenizer, 
    MBartForConditionalGeneration, 
    MBartConfig, 
    BeamSearchScorer, 
    HammingDiversityLogitsProcessor, 
    MinLengthLogitsProcessor, 
    LogitsProcessorList,
    TemperatureLogitsWarper
)
import torch
import numpy as np
import traceback

model = MBartForConditionalGeneration.from_pretrained(
    'facebook/mbart-large-cc25'
).cuda()

tokenizer = MBartTokenizer.from_pretrained(
    'facebook/mbart-large-cc25'
)

example = ''' <s> Dwell on the beauty of life. Watch the stars, and see yourself running with them. '''

input_ids = tokenizer([example], max_length=1024, return_tensors='pt')[
    'input_ids'].cuda()


# Group Beam Search test
num_beams = 24
num_beam_groups = 6

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

generate_args = {
    "input_ids": input_ids, 
    "min_length": 20,
    "num_beams": num_beams, 
    "no_repeat_ngram_size": 3,
    "decoder_start_token_id": 1, 
    "num_return_sequences": 24,
    "num_beam_groups": num_beam_groups,
    "diversity_penalty": 0.9,
    "use_cache": True
}

f = MBartForConditionalGeneration.group_beam_search

def g(self, input_ids, beam_scorer, **kwargs):
    print(kwargs)
    return f(self, input_ids, beam_scorer, **kwargs)

MBartForConditionalGeneration.group_beam_search = g

outputs = model.generate(
    **generate_args
    )


for sentence in tokenizer.batch_decode(outputs, skip_special_tokens=True):
    print(sentence)