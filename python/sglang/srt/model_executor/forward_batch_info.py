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
"""
Store information about a forward batch.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.utils import get_compiler_backend, is_hpu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ImageInputs, ModelWorkerBatch
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_PAD_SLOT_ID = 0
_PAD_BLOCK_ID = 0


class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers wil be IDLE if no sequence are allocated.
    IDLE = auto()

    # Used in speculative decoding: verify a batch in the target model.
    TARGET_VERIFY = auto()
    # Used in speculative decoding: extend a batch in the draft model.
    DRAFT_EXTEND = auto()

    # A dummy first batch to start the pipeline for overlap scheduler.
    # It is now used for triggering the sampling_info_done event for the first prefill batch.
    DUMMY_FIRST = auto()

    def is_prefill(self):
        return self == ForwardMode.PREFILL

    def is_extend(self):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or self == ForwardMode.DRAFT_EXTEND
            or self == self.TARGET_VERIFY
        )

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED

    def is_idle(self):
        return self == ForwardMode.IDLE

    def is_target_verify(self):
        return self == ForwardMode.TARGET_VERIFY

    def is_draft_extend(self):
        return self == ForwardMode.DRAFT_EXTEND

    def is_cuda_graph(self):
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
        )

    def is_dummy_first(self):
        return self == ForwardMode.DUMMY_FIRST

    def is_decode_or_idle(self):
        return self == ForwardMode.DECODE or self == ForwardMode.IDLE


class CaptureHiddenMode(IntEnum):
    NULL = auto()
    # Capture hidden states of all tokens.
    FULL = auto()
    # Capture a hidden state of the last token.
    LAST = auto()

    def need_capture(self):
        return self != CaptureHiddenMode.NULL

    def is_full(self):
        return self == CaptureHiddenMode.FULL

    def is_last(self):
        return self == CaptureHiddenMode.LAST

import itertools
import math
from vllm_hpu_extension.bucketing import find_bucket
from vllm_hpu_extension.ops import batch2block, block2batch

from vllm.utils import make_tensor_with_pad
def flatten(in_list):
    return list(itertools.chain(*in_list))

def make_cpu_tensor(data, max_len, pad, dtype, flat):
    if flat:
        data = [flatten(data)]
    result = make_tensor_with_pad(data,
                                  max_len=max_len,
                                  pad=pad,
                                  dtype=dtype,
                                  device='cpu')
    return result

def gather_list(tensor, indices, pad_value):
    result = [pad_value] * len(indices)
    for i, idx in enumerate(indices):
        if idx is not None:
            result[i] = tensor[idx]
    return result


def round_up(value: int, k: int) -> int:
    return (value + k - 1) // k * k


def pad_list(input, k, v):
    input_len = len(input)
    target_len = round_up(input_len, k)
    padding = target_len - input_len
    return input + [v] * padding


USE_CONTIGUOUS_PA = True


@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # For logits and logprobs post processing
    temp_scaled_logprobs: bool = False
    temperature: torch.Tensor = None
    top_p_normalized_logprobs: bool = False
    top_p: torch.Tensor = None

    # Position information
    positions: torch.Tensor = None

    # For decode
    decode_seq_lens_cpu: Optional[torch.Tensor] = None

    # For extend
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
    extend_prefix_lens_cpu: Optional[List[int]] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None
    extend_input_logprob_token_ids_gpu: Optional[torch.Tensor] = None

    # For multimodal
    image_inputs: Optional[List[ImageInputs]] = None

    # Encoder-decoder
    encoder_cached: Optional[List[bool]] = None
    encoder_lens: Optional[torch.Tensor] = None
    encoder_lens_cpu: Optional[List[int]] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # For LoRA
    lora_paths: Optional[List[str]] = None

    # For input embeddings
    input_embeds: Optional[torch.tensor] = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None

    # Attention backend
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: KVCache = None
    attn_backend: AttentionBackend = None

    # For DP attention
    global_num_tokens_cpu: Optional[List[int]] = None
    global_num_tokens_gpu: Optional[torch.Tensor] = None
    # Has to be None when cuda graph is captured.
    global_num_tokens_for_logprob_cpu: Optional[List[int]] = None
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] = None
    # for extend, local start pos and num tokens is different in logits processor
    # this will be computed in get_dp_local_info
    # this will be recomputed in LogitsMetadata.from_forward_batch
    dp_local_start_pos: Optional[torch.Tensor] = None  # cached info at runtime
    dp_local_num_tokens: Optional[torch.Tensor] = None  # cached info at runtime
    gathered_buffer: Optional[torch.Tensor] = None
    can_run_dp_cuda_graph: bool = False

    # Speculative decoding
    spec_info: Optional[Union[EagleVerifyInput, EagleDraftInput]] = None
    spec_algorithm: SpeculativeAlgorithm = None
    capture_hidden_mode: CaptureHiddenMode = None

    # For padding
    padded_static_len: int = -1  # -1 if not padded

    # For Qwen2-VL
    mrope_positions: torch.Tensor = None

    attn_bias: Optional[torch.Tensor] = None

    valid_seq_len: Optional[torch.Tensor] = None

    page_size: Optional[int] = None

    # For HPU paged attention
    block_list: Optional[torch.Tensor] = None
    block_mapping: Optional[torch.Tensor] = None
    block_usage: Optional[torch.Tensor] = None
    block_scales: Optional[torch.Tensor] = None
    block_groups: Optional[torch.Tensor] = None

    real_batch_size: Optional[int] = None

    @classmethod
    def _set_block_mapping(cls, metadata, batch_size, device, dtype):
        """Set block mapping using one-hot encoding of block groups."""

        mask = torch.arange(0,
                            metadata.page_size,
                            device=device,
                            dtype=torch.int32).unsqueeze(0)
        mask = mask >= metadata.block_usage.unsqueeze(-1)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask, -math.inf))

        # Handle out of bounds classes on CPU
        block_groups = metadata.block_groups.to(torch.long)
        block_mapping = torch.nn.functional.relu(block_groups)
        block_mapping = torch.nn.functional.one_hot(block_mapping, num_classes=batch_size)
        oob_values = block_groups.lt(0)
        block_mapping.masked_fill_(oob_values.unsqueeze(-1), 0)
        block_groups.masked_fill_(oob_values, batch_size)
        return block_mapping.to(dtype), attn_bias, block_groups

    @classmethod
    def _set_block_scales(cls, metadata, device):
        """Set block scales using batch2block and block2batch operations."""
        block_mapping = metadata.block_mapping
        ones = torch.ones((block_mapping.size(0),), device=device, dtype=block_mapping.dtype)
        sums = batch2block(block2batch(ones, block_mapping), block_mapping)
        block_scales = torch.reciprocal(torch.maximum(ones, sums))
        return block_scales

    @classmethod
    def _init_block_metadata(cls, ret, model_runner, block_tables, slot_mapping, block_size):
        """Initialize block metadata for HPU paged attention."""
        device = model_runner.device
        dtype = model_runner.dtype

        # Calculate block metadata
        last_block_usage = [
            slot % block_size + 1 for slot in slot_mapping
        ]
        block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
        block_usage = [[block_size] * (len(bt) - 1) + [lbu]
                      for bt, lbu in zip(block_tables, last_block_usage)
                      if bt]
        block_list = flatten(block_tables)
        block_groups = flatten(block_groups)
        block_usage = flatten(block_usage)
        assert len(block_list) == len(block_groups)
        assert len(block_list) == len(block_usage)

        if USE_CONTIGUOUS_PA:
            # Pad block metadata if needed
            block_bucket_size = max(max(block_list) + 1, len(block_list))
            block_bucket_size = find_bucket(block_bucket_size, (128, 128, 2048))
            indices = [None] * block_bucket_size
            for i, bid in enumerate(block_list):
                indices[bid] = i
            padding_fn = lambda tensor, pad_value: gather_list(tensor, indices, pad_value)
        else:
            block_bucket_size = find_bucket(len(block_list), (128, 128, 2048))
            padding_fn = lambda tensor, pad_value: pad_list(tensor, block_bucket_size, pad_value)

        block_list = padding_fn(block_list, _PAD_BLOCK_ID)
        block_groups = padding_fn(block_groups, -1)
        block_usage = padding_fn(block_usage, 1)

        # Convert to tensors
        ret.block_list = torch.tensor(block_list, dtype=torch.long, device=device)
        ret.block_groups = torch.tensor(block_groups, dtype=torch.long, device=device)
        ret.block_usage = torch.tensor(block_usage, dtype=dtype, device=device)

        # Set block mapping and scales
        ret.block_mapping, ret.attn_bias, ret.block_groups = cls._set_block_mapping(ret, ret.input_ids.shape[0], device, dtype)
        ret.block_scales = cls._set_block_scales(ret, device)


    @classmethod
    def init_new(
        cls,
        batch: ModelWorkerBatch,
        model_runner: ModelRunner,
    ):
        device = model_runner.device
        extend_input_logprob_token_ids_gpu = None
        if batch.extend_input_logprob_token_ids is not None:
            extend_input_logprob_token_ids_gpu = (
                batch.extend_input_logprob_token_ids.to(device, non_blocking=True)
            )
        ret = cls(
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            image_inputs=batch.image_inputs,
            encoder_cached=batch.encoder_cached,
            encoder_lens=batch.encoder_lens,
            encoder_lens_cpu=batch.encoder_lens_cpu,
            encoder_out_cache_loc=batch.encoder_out_cache_loc,
            seq_lens_sum=batch.seq_lens_sum,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
            token_ids_logprobs=batch.token_ids_logprobs,
            can_run_dp_cuda_graph=batch.can_run_dp_cuda_graph,
            lora_paths=batch.lora_paths,
            sampling_info=batch.sampling_info,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            attn_backend=model_runner.attn_backend,
            spec_algorithm=batch.spec_algorithm,
            spec_info=batch.spec_info,
            capture_hidden_mode=batch.capture_hidden_mode,
            input_embeds=batch.input_embeds,
            extend_input_logprob_token_ids_gpu=extend_input_logprob_token_ids_gpu,
        )

        # For DP attention
        if batch.global_num_tokens is not None:
            ret.global_num_tokens_cpu = batch.global_num_tokens
            ret.global_num_tokens_gpu = torch.tensor(
                batch.global_num_tokens, dtype=torch.int64
            ).to(device, non_blocking=True)

            ret.global_num_tokens_for_logprob_cpu = batch.global_num_tokens_for_logprob
            ret.global_num_tokens_for_logprob_gpu = torch.tensor(
                batch.global_num_tokens_for_logprob, dtype=torch.int64
            ).to(device, non_blocking=True)

            sum_len = sum(batch.global_num_tokens)
            ret.gathered_buffer = torch.zeros(
                (sum_len, model_runner.model_config.hidden_size),
                dtype=model_runner.dtype,
                device=device,
            )
        if ret.forward_mode.is_idle():
            ret.positions = torch.empty((0,), device=device)
            return ret

        # Override the positions with spec_info
        if (
            ret.spec_info is not None
            and getattr(ret.spec_info, "positions", None) is not None
        ):
            ret.positions = ret.spec_info.positions

        # Init position information
        if ret.forward_mode.is_decode():
            if ret.positions is None:
                ret.positions = clamp_position(batch.seq_lens)
            if ret.decode_seq_lens_cpu is None:
                ret.decode_seq_lens_cpu = batch.decode_seq_lens
        else:
            ret.extend_seq_lens = torch.tensor(
                batch.extend_seq_lens, dtype=torch.int32
            ).to(device, non_blocking=True)
            ret.extend_prefix_lens = torch.tensor(
                batch.extend_prefix_lens, dtype=torch.int32
            ).to(device, non_blocking=True)
            if model_runner.server_args.attention_backend not in ["torch_native", "hpu"]:
                ret.extend_num_tokens = batch.extend_num_tokens
                positions, ret.extend_start_loc = compute_position_triton(
                    ret.extend_prefix_lens,
                    ret.extend_seq_lens,
                    ret.extend_num_tokens,
                )
            else:
                positions, ret.extend_start_loc = compute_position_torch(
                    ret.extend_prefix_lens, ret.extend_seq_lens
                )
            if ret.positions is None:
                ret.positions = positions
            ret.extend_prefix_lens_cpu = batch.extend_prefix_lens
            ret.extend_seq_lens_cpu = batch.extend_seq_lens
            ret.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens

        if model_runner.model_is_mrope:
            ret._compute_mrope_positions(model_runner, batch)

        # Init lora information
        if model_runner.server_args.lora_paths is not None:
            model_runner.lora_manager.prepare_lora_batch(ret)

        seq_len_list = ret.extend_seq_lens_cpu
        if model_runner.server_args.attention_backend == "hpu":
            ret.page_size = model_runner.token_to_kv_pool_allocator.page_size
            if ret.forward_mode.is_extend():
                sum_seq_len = sum(seq_len_list)
                max_prompt_len = find_bucket(sum_seq_len, (128, 128, 2048))
                ret.attn_bias = cls.make_hpu_attn_bias(
                    seq_lens=seq_len_list,
                    max_prompt_len=max_prompt_len,
                    dtype=model_runner.dtype,
                ).to(model_runner.device)
                padding_len = max_prompt_len - sum_seq_len
                ret.input_ids = torch.nn.functional.pad(ret.input_ids, (0, padding_len), value=0)
                ret.positions = torch.nn.functional.pad(ret.positions, (0, padding_len), value=0)
                ret.valid_seq_len = torch.tensor(sum_seq_len, dtype=torch.int32)
                ret.out_cache_loc = torch.nn.functional.pad(ret.out_cache_loc, (0, padding_len), value=0)
                ret.real_batch_size = ret.batch_size
                ret.batch_size = 1
            else:
                # Initialize block metadata for HPU paged attention
                from sglang.srt.mem_cache.paged_allocator import HPUPagedTokenToKVPoolAllocator
                paged_allocator: HPUPagedTokenToKVPoolAllocator = model_runner.token_to_kv_pool_allocator
                padded_batch_size = find_bucket(ret.batch_size, (1, 32, 128))
                block_tables = []
                for i in range(ret.batch_size):
                    block_tables.append(paged_allocator.block_manager.seq_info[ret.req_pool_indices[i].item()][0])

                for i in range(padded_batch_size - ret.batch_size):
                    block_tables.append([_PAD_BLOCK_ID])

                ret.input_ids = torch.nn.functional.pad(ret.input_ids, (0, padded_batch_size), value=0)
                ret.positions = torch.nn.functional.pad(ret.positions, (0, padded_batch_size), value=0)
                ret.valid_seq_len = torch.ones(padded_batch_size, dtype=torch.int32)
                ret.out_cache_loc = torch.nn.functional.pad(ret.out_cache_loc, (0, padded_batch_size), value=0)
                ret.real_batch_size = ret.batch_size
                ret.batch_size = padded_batch_size
                slot_mapping = ret.out_cache_loc
                block_size = paged_allocator.page_size

                cls._init_block_metadata(ret, model_runner, block_tables, slot_mapping, block_size)

        return ret

    @classmethod
    def make_hpu_attn_bias(cls, seq_lens, max_prompt_len, dtype):
        seq_pos = [list(range(sl)) for sl in seq_lens]
        seq_idx = [[i] * sl for i, sl in enumerate(seq_lens)]
        seq_pos = make_cpu_tensor(seq_pos,
                                  max_len=max_prompt_len,
                                  pad=-1,
                                  dtype=torch.long,
                                  flat=True)
        seq_idx = make_cpu_tensor(seq_idx,
                                  max_len=max_prompt_len,
                                  pad=-1,
                                  dtype=torch.long,
                                  flat=True)
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

    def get_merged_image_inputs(self) -> Optional[ImageInputs]:
        """
        Merge all image inputs in the batch into a single ImageInputs object.

        Returns:
            if none, current batch contains no image input

        """
        if not self.image_inputs or all(x is None for x in self.image_inputs):
            return None

        # Filter out None values
        valid_inputs = [x for x in self.image_inputs if x is not None]

        # Start with the first valid image input
        merged = valid_inputs[0]

        # Merge remaining inputs
        for img_input in valid_inputs[1:]:
            merged.merge(img_input)

        if isinstance(merged.pixel_values, np.ndarray):
            merged.pixel_values = torch.from_numpy(merged.pixel_values)

        return merged

    def _compute_mrope_positions(
        self, model_runner: ModelRunner, batch: ModelWorkerBatch
    ):
        device = model_runner.device
        hf_config = model_runner.model_config.hf_config
        mrope_positions_list = [None] * self.seq_lens.shape[0]
        if self.forward_mode.is_decode():
            for i, _ in enumerate(mrope_positions_list):
                mrope_position_delta = (
                    0
                    if batch.image_inputs[i] is None
                    else batch.image_inputs[i].mrope_position_delta
                )
                mrope_positions_list[i] = MRotaryEmbedding.get_next_input_positions(
                    mrope_position_delta,
                    int(self.seq_lens[i]) - 1,
                    int(self.seq_lens[i]),
                )
        elif self.forward_mode.is_extend():
            extend_start_loc_cpu = self.extend_start_loc.cpu().numpy()
            for i, image_inputs in enumerate(batch.image_inputs):
                extend_start_loc, extend_seq_len, extend_prefix_len = (
                    extend_start_loc_cpu[i],
                    batch.extend_seq_lens[i],
                    batch.extend_prefix_lens[i],
                )
                if image_inputs is None:
                    # text only
                    mrope_positions = [
                        [
                            pos
                            for pos in range(
                                extend_prefix_len, extend_prefix_len + extend_seq_len
                            )
                        ]
                    ] * 3
                else:
                    # TODO: current qwen2-vl do not support radix cache since mrope position calculation
                    mrope_positions, mrope_position_delta = (
                        MRotaryEmbedding.get_input_positions(
                            input_tokens=self.input_ids[
                                extend_start_loc : extend_start_loc + extend_seq_len
                            ],
                            image_grid_thw=image_inputs.image_grid_thws,
                            vision_start_token_id=hf_config.vision_start_token_id,
                            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
                            context_len=0,
                        )
                    )
                    batch.image_inputs[i].mrope_position_delta = mrope_position_delta
                mrope_positions_list[i] = mrope_positions

        self.mrope_positions = torch.cat(
            [torch.tensor(pos, device=device) for pos in mrope_positions_list],
            axis=1,
        )
        self.mrope_positions = self.mrope_positions.to(torch.int64)


def compute_position_triton(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, extend_seq_lens_sum
):
    """Compute positions. It is a fused version of `compute_position_torch`."""
    batch_size = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch_size

    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    extend_start_loc = torch.empty(
        batch_size, dtype=torch.int32, device=extend_seq_lens.device
    )

    # Launch kernel
    compute_position_kernel[(batch_size,)](
        positions,
        extend_start_loc,
        extend_prefix_lens,
        extend_seq_lens,
        has_prefix,
    )

    return positions, extend_start_loc


@triton.jit
def compute_position_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
    has_prefix: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0).to(tl.int64)

    prefix_len = tl.load(extend_prefix_lens + pid) if has_prefix else 0
    seq_len = tl.load(extend_seq_lens + pid)

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_seq_lens + i)

    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        tl.store(
            positions + cumsum_start + offset,
            prefix_len + offset,
            mask=offset < seq_len,
        )
    tl.store(extend_start_loc + pid, cumsum_start)


def compute_position_torch(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor
):
    positions = torch.cat(
        [
            torch.arange(
                prefix_len, prefix_len + extend_len, device=extend_prefix_lens.device
            )
            for prefix_len, extend_len in zip(extend_prefix_lens, extend_seq_lens)
        ],
        axis=0,
    )
    extend_start_loc = torch.zeros_like(extend_seq_lens)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    return positions.to(torch.int64), extend_start_loc


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=is_hpu())
def clamp_position(seq_lens):
    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)
