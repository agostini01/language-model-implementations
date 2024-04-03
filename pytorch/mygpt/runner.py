import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        res = []

        # Context is B x T
        # len(context) is B, len(context.T) is T
        for i in range(new_chars):

            # Only allow the last tokens in the context
            if len(context.T) > context_length:
                context = context[:, -context_length:]

            prediction = model(context)  # B x T x V
            last_time_step = prediction[:, -1, :]  # B x V
            probs = nn.functional.softmax(last_time_step, dim=-1)

            next_token = torch.multinomial(probs, 1, generator=generator)
            generator.set_state(initial_state)
            context = torch.cat([context, next_token], dim=-1)  # B x (T+1)
            res.append(int_to_char[next_token.item()])
        return ''.join(res)
