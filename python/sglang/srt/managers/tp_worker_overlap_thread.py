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
"""A tensor parallel worker."""

import dataclasses
import logging
import signal
import threading
from queue import Queue
from typing import Optional

import psutil
import torch

from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import DynamicGradMode, get_compiler_backend, is_hpu
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=is_hpu())
def resolve_future_token_ids(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )

_is_hpu = is_hpu()

_PAD_SLOT_ID = 0
_PAD_BLOCK_ID = 0

PREFILL_BUCKET_MIN = 512
PREFILL_BUCKET_STEP = 512
PREFILL_BUCKET_MAX = 4096

DECODE_BLOCK_BUCKET_MIN = 128
DECODE_BLOCK_BUCKET_STEP = 128
DECODE_BLOCK_BUCKET_MAX = 4096

DECODE_BATCH_BUCKET_MIN = 1
DECODE_BATCH_BUCKET_STEP = 32
DECODE_BATCH_BUCKET_MAX = 192

import itertools
import math
import os
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


def _set_block_mapping(metadata, batch_size, device, dtype):
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

def _set_block_scales(metadata, device):
    """Set block scales using batch2block and block2batch operations."""
    block_mapping = metadata.block_mapping
    ones = torch.ones((block_mapping.size(0),), device=device, dtype=block_mapping.dtype)
    sums = batch2block(block2batch(ones, block_mapping), block_mapping)
    block_scales = torch.reciprocal(torch.maximum(ones, sums))
    return block_scales


def _init_block_metadata(ret, model_runner, block_tables, slot_mapping, block_size):
    """Initialize block metadata for HPU paged attention."""
    device = "cpu"
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

    if ret.use_contiguous_pa:
        # Pad block metadata if needed
        block_bucket_size = max(max(block_list) + 1, len(block_list))
        block_bucket_size = find_bucket(block_bucket_size, (DECODE_BLOCK_BUCKET_MIN, DECODE_BLOCK_BUCKET_STEP, DECODE_BLOCK_BUCKET_MAX))
        indices = [None] * block_bucket_size
        for i, bid in enumerate(block_list):
            indices[bid] = i
        padding_fn = lambda tensor, pad_value: gather_list(tensor, indices, pad_value)
    else:
        block_bucket_size = find_bucket(len(block_list), (DECODE_BLOCK_BUCKET_MIN, DECODE_BLOCK_BUCKET_STEP, DECODE_BLOCK_BUCKET_MAX))
        padding_fn = lambda tensor, pad_value: pad_list(tensor, block_bucket_size, pad_value)

    block_list = padding_fn(block_list, _PAD_BLOCK_ID)
    block_groups = padding_fn(block_groups, -1)
    block_usage = padding_fn(block_usage, 1)

    # Convert to tensors
    ret.block_list = torch.tensor(block_list, dtype=torch.long, device=device)
    ret.block_groups = torch.tensor(block_groups, dtype=torch.long, device=device)
    ret.block_usage = torch.tensor(block_usage, dtype=dtype, device=device)

    # Set block mapping and scales
    ret.block_mapping, ret.attn_bias, ret.block_groups = _set_block_mapping(ret, ret.batch_size, device, dtype)
    ret.block_scales = _set_block_scales(ret, device)

def make_hpu_attn_bias(seq_lens, max_prompt_len, dtype):
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
    # q_seq_idx = seq_idx.unsqueeze(-1)
    # kv_seq_idx = seq_idx.unsqueeze(-2)
    # q_seq_pos = seq_pos.unsqueeze(-1)
    # kv_seq_pos = seq_pos.unsqueeze(-2)
    # seq_idx = q_seq_idx != kv_seq_idx
    # seq_pos = kv_seq_pos > q_seq_pos
    # attn_mask = seq_idx | seq_pos
    
    # attn_bias.masked_fill_(attn_mask, -math.inf)
    # return attn_bias.unsqueeze(1)
    attn_bias = torch.zeros(1, 1, max_prompt_len, max_prompt_len, dtype=dtype)
    return attn_bias, seq_pos, seq_idx

from sglang.srt.model_executor.forward_batch_info import clamp_position, compute_position_triton, compute_position_torch

def create_hpu_specific_fields(ret: ModelWorkerBatch, model_runner):


    if ret.forward_mode.is_decode():
        if ret.positions is None:
            ret.positions = clamp_position(ret.seq_lens)
    else:
        ret.extend_seq_lens = torch.tensor(
            ret.extend_seq_lens, dtype=torch.int32
        ).to("cpu", non_blocking=True)
        ret.extend_prefix_lens = torch.tensor(
            ret.extend_prefix_lens, dtype=torch.int32
        ).to("cpu", non_blocking=True)
        if model_runner.server_args.attention_backend not in ["torch_native", "hpu"]:
            ret.extend_num_tokens = ret.extend_num_tokens
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

    ret.page_size = model_runner.token_to_kv_pool_allocator.page_size
    if ret.forward_mode.is_extend():
        seq_len_list = ret.extend_seq_lens
        sum_seq_len = sum(seq_len_list)
        max_prompt_len = find_bucket(sum_seq_len, (PREFILL_BUCKET_MIN, PREFILL_BUCKET_STEP, PREFILL_BUCKET_MAX))
        ret.attn_bias, ret.seq_pos, ret.seq_idx = make_hpu_attn_bias(
            seq_lens=seq_len_list,
            max_prompt_len=max_prompt_len,
            dtype=model_runner.dtype,
        )
        padding_len = max_prompt_len - sum_seq_len
        max_prefill_seqs = model_runner.server_args.max_running_requests
        batch_size = len(seq_len_list)
        ret.input_ids = torch.nn.functional.pad(ret.input_ids, (0, padding_len), value=0)
        ret.positions = torch.nn.functional.pad(ret.positions, (0, padding_len), value=0)
        ret.valid_seq_len = torch.tensor(sum_seq_len, dtype=torch.int32)
        ret.extend_seq_lens_padded = torch.nn.functional.pad(ret.extend_seq_lens, (0, max_prefill_seqs - batch_size), value=0)
        ret.out_cache_loc = torch.nn.functional.pad(ret.out_cache_loc, (0, padding_len), value=0)
        ret.real_batch_size = len(ret.seq_lens)
        ret.batch_size = 1
    else:
        ret.use_contiguous_pa = os.environ.get('SGLANG_HPU_CONTIGUOUS_PA',
                                        'true').lower() in ['true', '1']
        batch_size = len(ret.seq_lens)
        page_size = model_runner.token_to_kv_pool_allocator.page_size
        # Initialize block metadata for HPU paged attention
        from sglang.srt.managers.schedule_batch import ReqToTokenPool
        req_token_pool: ReqToTokenPool = model_runner.req_to_token_pool
        padded_batch_size = find_bucket(batch_size, (DECODE_BATCH_BUCKET_MIN, DECODE_BATCH_BUCKET_STEP, DECODE_BATCH_BUCKET_MAX))
        block_tables = []
        slots_list = []
        for i in range(batch_size):
            slots = req_token_pool.req_to_token[ret.req_pool_indices[i], :ret.seq_lens[i]]
            last_loc = slots[-1]
            num_full_tables = (ret.seq_lens[i] - 1) // page_size
            ranges = torch.arange(0, num_full_tables*page_size, step=page_size, device=ret.input_ids.device)
            pages = slots[ranges] // page_size
            pages = pages.flatten().tolist()
            if last_loc % page_size != 0:
                pages.append((last_loc // page_size).item())
            block_tables.append(pages)
            slots_list.append(slots)
        for i in range(padded_batch_size - batch_size):
            block_tables.append([_PAD_BLOCK_ID])
        print(f"ret.out_cache_loc: {ret.out_cache_loc}")

        padding_len = padded_batch_size - len(ret.seq_lens)
        input_ids = torch.nn.functional.pad(ret.input_ids, (0, padding_len), value=0)
        positions = torch.nn.functional.pad(ret.positions, (0, padding_len), value=0)
        ret.valid_seq_len = torch.ones(padded_batch_size, dtype=torch.int32)
        ret.out_cache_loc = torch.nn.functional.pad(ret.out_cache_loc, (0, padding_len), value=0)
        ret.real_batch_size = len(ret.seq_lens)
        
        slot_mapping = ret.out_cache_loc
        ret.input_ids = input_ids
        ret.positions = positions
        ret.batch_size = padded_batch_size
        _init_block_metadata(ret, model_runner, block_tables, slot_mapping, page_size)
    return ret

class TpModelWorkerClientSingelThread:

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
    ):
        # Load the model
        self.worker = TpModelWorker(server_args, gpu_id, tp_rank, dp_rank, nccl_port)
    
    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
    ):
        create_hpu_specific_fields(model_worker_batch, self.worker.model_runner)
        return self.worker.forward_batch_generation(model_worker_batch, launch_done, skip_sample)
    
    def __getattr__(self, name):
        return getattr(self.worker, name)
        


class TpModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
    ):
        # Load the model
        self.worker = TpModelWorker(server_args, gpu_id, tp_rank, dp_rank, nccl_port)
        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device
        self.gpu_id = gpu_id

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * 3
        self.future_token_ids_map = torch.empty(
            (self.max_running_requests * 5,), dtype=torch.int64, device=self.device
        )

        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.forward_stream = torch.get_device_module(self.device).Stream()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()
        self.scheduler_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.scheduler_stream.synchronize = lambda: None  # No-op for CPU

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_tp_cpu_group(self):
        return self.worker.get_tp_cpu_group()

    def get_attention_tp_cpu_group(self):
        return self.worker.get_attention_tp_cpu_group()

    def get_memory_pool(self):
        return (
            self.worker.model_runner.req_to_token_pool,
            self.worker.model_runner.token_to_kv_pool_allocator,
        )

    def get_kv_cache(self):
        return self.worker.model_runner.token_to_kv_pool

    def forward_thread_func(self):
        try:
            with torch.get_device_module(self.device).stream(self.forward_stream):
                self.forward_thread_func_() 
        except Exception:
            traceback = get_exception_traceback()
            logger.error(f"TpModelWorkerClient hit an exception: {traceback}")
            self.parent_process.send_signal(signal.SIGQUIT)

    @DynamicGradMode()
    def forward_thread_func_(self):
        batch_pt = 0
        batch_lists = [None] * 2

        while True:
            model_worker_batch, future_token_ids_ct = self.input_queue.get()
            if not model_worker_batch:
                break

            # Keep a reference of model_worker_batch by storing it into a list.
            # Otherwise, the tensor members of model_worker_batch will be released
            # by pytorch and cause CUDA illegal memory access errors.
            batch_lists[batch_pt % 2] = model_worker_batch
            batch_pt += 1

            # Create event
            self.launch_done = threading.Event()
            if not _is_hpu:
                copy_done = torch.get_device_module(self.device).Event()
            else:
                copy_done = None

            # Resolve future tokens in the input
            input_ids = model_worker_batch.input_ids
            resolve_future_token_ids(input_ids, self.future_token_ids_map)

            # Run forward
            logits_output, next_token_ids = self.worker.forward_batch_generation(
                model_worker_batch, self.launch_done
            )

            # Update the future token ids map
            bs = len(model_worker_batch.seq_lens)
            self.future_token_ids_map[
                future_token_ids_ct + 1 : future_token_ids_ct + bs + 1
            ] = next_token_ids

            # Copy results to the CPU
            if model_worker_batch.return_logprob:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.to("cpu", non_blocking=True)
                )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = (
                        logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                    )
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states.to(
                    "cpu", non_blocking=True
                )
            next_token_ids = next_token_ids.to("cpu", non_blocking=True)
            if copy_done is not None:
                copy_done.record()

            self.output_queue.put((copy_done, logits_output, next_token_ids))

    def resolve_batch_result(self, bid: int):
        copy_done, logits_output, next_token_ids = self.output_queue.get()
        if copy_done is not None:
            copy_done.synchronize()
        self.launch_done.wait()

        if logits_output.next_token_logprobs is not None:
            logits_output.next_token_logprobs = (
                logits_output.next_token_logprobs.tolist()
            )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )
        next_token_ids = next_token_ids.tolist()
        return logits_output, next_token_ids

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

        create_hpu_specific_fields(model_worker_batch, self.worker.model_runner)

        # A cuda stream sync here to avoid the cuda illegal memory access error.
        if not _is_hpu:
            self.scheduler_stream.synchronize()

        # Push a new batch to the queue
        self.input_queue.put((model_worker_batch, self.future_token_ids_ct))

        # Allocate output future objects
        bs = len(model_worker_batch.seq_lens)
        future_next_token_ids = torch.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=torch.int64,
            device="cpu",
        )
        self.future_token_ids_ct = (
            self.future_token_ids_ct + bs
        ) % self.future_token_ids_limit
        return None, future_next_token_ids

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.worker.update_weights_from_disk(recv_req)
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.worker.init_weights_update_group(recv_req)
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.worker.update_weights_from_distributed(recv_req)
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        success, message = self.worker.update_weights_from_tensor(recv_req)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        return self.worker.get_weights_by_name(recv_req)

    def __delete__(self):
        self.input_queue.put((None, None))
        self.copy_queue.put((None, None, None))
