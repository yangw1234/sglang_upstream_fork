# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the model with cuda graph and torch.compile."""

from __future__ import annotations


from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import tqdm
import math

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
)
if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

def make_hpu_attn_bias(seq_pos, seq_idx, dtype):
    q_seq_idx = seq_idx.unsqueeze(-1)
    kv_seq_idx = seq_idx.unsqueeze(-2)
    q_seq_pos = seq_pos.unsqueeze(-1)
    kv_seq_pos = seq_pos.unsqueeze(-2)
    seq_idx = q_seq_idx != kv_seq_idx
    seq_pos = kv_seq_pos > q_seq_pos
    attn_mask = seq_idx | seq_pos
    attn_bias = torch.zeros_like(attn_mask, dtype=dtype)
    attn_bias.masked_fill_(attn_mask, -math.inf)
    return attn_bias.unsqueeze(1)
    

class HPUAdapter:

    def __init__(self, model, dtype) -> None:
        self.model = model
        self.dtype = dtype
    

    def __getattr__(self, name):
        return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        assert len(args) == 3, "Only three arguments are supported"
        input_batch = args[2]
        if input_batch.forward_mode.is_extend():
            input_batch.attn_bias.copy_(make_hpu_attn_bias(input_batch.seq_pos, input_batch.seq_idx, self.dtype))
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class HPUGraphRunner:
    """A HPUGraphRunner runs the forward pass of a model with HPU graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        import habana_frameworks.torch as htorch
        self.model = htorch.hpu.wrap_in_hpu_graph(
            HPUAdapter(self.model_runner.model, self.model_runner.dtype),
            disable_tensor_cache=True,
        ) if htorch.utils.internal.is_lazy() else HPUAdapter(self.model_runner.model, self.model_runner.dtype)
        # Capture
        try:
            with self.model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture hpu graph failed: {e}\n"
            )

    @contextmanager
    def model_capture_mode(self):
        yield

    def can_run(self, forward_batch: ForwardBatch):
        return True

    def capture(self):
        # TODO: implement warmup
        pass

    def replay(
        self, forward_batch: ForwardBatch, skip_attn_backend_init: bool = False
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        
        logits_output = self.model.forward(forward_batch.input_ids, forward_batch.positions, forward_batch)

        logits_output = LogitsProcessorOutput(
            next_token_logits=logits_output.next_token_logits.clone(),
            hidden_states=logits_output.hidden_states.clone() if logits_output.hidden_states is not None else None,
        )
        return logits_output