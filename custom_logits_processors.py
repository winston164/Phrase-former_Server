from transformers import LogitsProcessor
import torch
import random


def make_blocking_dict(tokens, p=0.5):
    res = dict()
    for i, token in enumerate(tokens):
        if len(tokens) > (i + 1):
            res[token.item()] = tokens[i + 1].item()
    for key in list(res.keys()):
        if random.random() > p:
            res.pop(key)

    return res


class DynamicBlockingProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` to block specific words according to dynamic blocking dictionary, meant for group_beam_search.

    Args:
        min_length (:obj:`int`):
            The minimum length below which the score of :obj:`eos_token_id` is set to :obj:`-float("Inf")`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, input_ids: torch.LongTensor, num_beams: int, num_beam_groups: int, blocking_percent: float):
        assert type(num_beam_groups) is int and type(
            num_beams) is int and num_beams > num_beam_groups, "For DynamicBlockingProcessor num_beams and num_beam_groups must be integers and num_beams>num_beam_groups"
        assert blocking_percent > 0.0 and blocking_percent < 1.0, "For DynamicBlockingProcessor the blocking_percent most be a float between 0 and 1"

        self._num_beams = num_beams
        self._num_sub_beams = num_beams // num_beam_groups
        self._num_batches = input_ids.shape[0] 

        # Each batch has a blocking dictionary for each beam group
        self.blocking_dicts = [[make_blocking_dict(batch_ids, blocking_percent) for _ in range(
            num_beam_groups)] for batch_ids in input_ids]

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int
    ) -> torch.FloatTensor:
        # Separate batches
        for batch_idx in range(self._num_batches):
          # For each batch get the group dictionary, score, and input_ids
          group_block_dict = self.blocking_dicts[batch_idx][beam_group_idx]
          group_scores = scores[batch_idx * self._num_sub_beams : (batch_idx + 1) * self._num_sub_beams]
          group_input_ids = input_ids[batch_idx * self._num_sub_beams : (batch_idx + 1) * self._num_sub_beams] 

          for i in range(self._num_sub_beams):
            # For each beam see the last token 
            last_id = group_input_ids[i][-1].item()

            # Check if it is in blocking dict
            if last_id in group_block_dict.keys():
              # If so, make the value of the blocked word -inf
              group_scores[i][group_block_dict[last_id]] = -float("inf")
            pass

          scores[batch_idx * self._num_sub_beams : (batch_idx + 1) * self._num_sub_beams] = group_scores


        return scores
