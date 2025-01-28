"""
Example script demonstrating inference on Habana Gaudi processors using vLLM's HPU worker.
"""

import os
import torch
import habana_frameworks.torch as htorch
from vllm.config import (
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    DeviceConfig,
    CacheConfig,
    VllmConfig,
    ConfigFormat
)
from vllm.worker.hpu_worker import HPUWorker
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata, SequenceData
from array import array
from typing import Optional, Dict, Any, List, Tuple, Union
from vllm.engine.arg_utils import EngineArgs, TaskOption
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.hpu_model_runner import ModelInputForHPUWithSamplingMetadata
from vllm.attention.backends.hpu_attn import HPUAttentionMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata, SequenceGroupToSample
from vllm.worker.hpu_worker import WorkerInput
from vllm.model_executor.sampling_metadata import SamplingType
from vllm.worker.hpu_model_runner import precompute_indices_and_offsets

from .hpu_memory_pool import BlockManager, HPUTokenToKVPool, HPUReqToTokenPool
from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput


class HPUTPWorker:

    def __init__(self, model_path: str, server_args: ServerArgs):

        self.vllm_engine_args = create_engine_args(model=model_path, device="hpu",
                                                   dtype="bfloat16",
                                                   block_size=128,
                                                   max_model_len=2048)
        self.vllm_config = self.vllm_engine_args.create_engine_config()
        self.distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())

        # Initialize HPU worker
        self.vllm_worker = HPUWorker(
            vllm_config=self.vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=self.distributed_init_method,
            is_driver_worker=True
        )
        self.vllm_worker.init_device()
        self.vllm_worker.load_model()
        self.tp_rank = self.vllm_worker.local_rank
        self.device = self.vllm_worker.device

        num_gpu_blocks, num_cpu_blocks = self.vllm_worker.determine_num_available_blocks()
        self.vllm_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        self.block_manager = BlockManager(
            block_size=self.vllm_config.cache_config.block_size,
            num_blocks=num_gpu_blocks
        )

        if server_args.max_running_requests is None:
            server_args.max_running_requests = 128

        self.req_to_token_pool = HPUReqToTokenPool(
            block_manager=self.block_manager,
            size=server_args.max_running_requests
        )   

        self.token_to_kv_pool = HPUTokenToKVPool(
            vllm_cache_engine=self.vllm_worker.cache_engine,
            block_manager=self.block_manager
        )
        self.max_total_num_tokens = num_gpu_blocks * self.vllm_config.cache_config.block_size
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = server_args.max_running_requests
        self.max_req_len = 2047
        self.max_req_input_len = self.max_req_len - 5
        self.random_seed = server_args.random_seed
    
    def get_memory_pool(self):
        return self.req_to_token_pool, self.token_to_kv_pool

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            None,
            None,
            None,
            None,
        )
    
    def forward_batch_generation(self, 
        model_worker_batch: ModelWorkerBatch,
        launch_done = None,
        skip_sample: bool = False,
    ):
        """Execute forward pass for a batch of requests."""
        print("\033[1;32;40m forward_batch_generation: \033[0m", model_worker_batch)
        
        # # Convert ModelWorkerBatch to HPU model input format
        if model_worker_batch.forward_mode.is_extend():
            model_input = self.create_model_input_from_batch_prefill(
                model_worker_batch,
                device=self.device
            )
        else:
            model_input = self.create_model_input_from_batch_decode(
                model_worker_batch,
                device=self.device
            )
        print("\033[1;34;40m model_input: \033[0m", model_input)
        # # Execute model forward pass
        output = self.vllm_worker.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.vllm_worker.kv_cache[0] if self.vllm_worker.kv_cache is not None else None,
            intermediate_tensors=None,
            num_steps=1,
            **{}
        )
        result = self.create_sgl_logits_output_from_vllm_sampler_output(output[0])
        print("\033[1;32;40m Forward Batch Generation Result: \033[0m", result)
        return result

    def create_sgl_logits_output_from_vllm_sampler_output(self, output):
        """Convert vLLM's sampler output to SGLang's LogitsProcessorOutput format.
        
        Args:
            output: vLLM's SamplerOutput containing sampled tokens and logprobs
            
        Returns:
            Tuple[LogitsProcessorOutput, List[int]]: SGLang's format for logits output and next token IDs
        """
        # Extract next token IDs
        next_token_ids = []
        for seq_output in output.outputs:
            for sample in seq_output.samples:
                next_token_ids.append(sample.output_token)

        # Create LogitsProcessorOutput
        logits_output = LogitsProcessorOutput(
            next_token_logits=None,
            # Other fields are optional and can be None
            hidden_states=None,
            next_token_logprobs=None, 
            next_token_top_logprobs_val=None,
            next_token_top_logprobs_idx=None,
            input_token_logprobs=None,
            input_top_logprobs_val=None,
            input_top_logprobs_idx=None
        )

        return logits_output, torch.tensor(next_token_ids, device="cpu", dtype=torch.int32)

    def create_model_input_from_batch_decode(
        self,
        model_worker_batch: ModelWorkerBatch,
        device: str = "hpu:0"
    ) -> ModelInputForHPUWithSamplingMetadata:
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=128
        )

        batch_size = len(model_worker_batch.req_pool_indices)
        input_tokens = model_worker_batch.input_ids.to(device).reshape(batch_size, 1)
        input_positions = model_worker_batch.seq_lens.to(device).reshape(batch_size, 1)
        seq_lens = []
        query_lens = []
        lora_mapping = None
        lora_requests = set()
        block_tables = []
        for i in range(batch_size):
            block_tables.append(self.block_manager.seq_info[model_worker_batch.req_pool_indices[i].item()][0])
        block_list, block_mapping, block_usage, block_scales = self.create_block_info_from_block_tables(block_tables,
                                                                                                        model_worker_batch.out_cache_loc,
                                                                                                        bucket_size=128,
                                                                                                        block_size=self.vllm_config.cache_config.block_size)
        slot_mapping = torch.tensor(model_worker_batch.out_cache_loc, device=device, dtype=torch.int32).reshape(batch_size, 1)
        block_indices, block_offsets = precompute_indices_and_offsets(
            self.vllm_config.cache_config.block_size,
            slot_mapping,
            is_prompt=False
        )
        attn_metadata = HPUAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=batch_size,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            block_list=block_list,
            block_mapping=block_mapping,
            block_usage=block_usage,
            block_indices=block_indices,
            block_offsets=block_offsets,
            block_scales=block_scales,
            is_prompt=False,
            attn_bias=None,
            seq_lens_tensor=None
        )

        seq_groups = []
        for i in range(batch_size):
            seq_groups.append(SequenceGroupToSample(
                seq_ids=[model_worker_batch.req_pool_indices[i]],
                seq_data={model_worker_batch.req_pool_indices[i]: SequenceData(
                    _prompt_token_ids=array('l', [])
                )},
                sampling_params=sampling_params,
                seq_len=None,
                query_len=1,
                generator=None,
                is_prompt=False,
                prompt_logprob_indices=[],
                sample_indices=[i]
            ))
        
        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            selected_token_indices=torch.arange(batch_size, device=device),
            categorized_sample_indices={
                SamplingType.GREEDY: torch.tensor([], device=device, dtype=torch.int32),
                SamplingType.RANDOM: torch.arange(batch_size, device=device, dtype=torch.int32),
                SamplingType.RANDOM_SEED: torch.tensor([], device=device, dtype=torch.int32)
            },
            num_prompts=0,
            skip_sampler_cpu_output=False,
            reuse_sampling_tensors=False
        )

        model_input = ModelInputForHPUWithSamplingMetadata(
            input_tokens=input_tokens,
            input_positions=input_positions,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_mapping=lora_mapping,
            lora_requests=lora_requests,
            attn_metadata=attn_metadata,
            multi_modal_kwargs={},
            real_batch_size=batch_size,
            batch_size_padded=batch_size,
            virtual_engine=0,
            lora_ids=[0] * batch_size,
            async_callback=None,
            sampling_metadata=sampling_metadata
        )

        return model_input

    def create_block_info_from_block_tables(self, block_tables,
                                            slot_mapping,
                                            bucket_size,
                                            block_size):
        from vllm.worker.hpu_model_runner import find_bucket, pad_list, _PAD_BLOCK_ID
        import itertools

        blocks_used = [len(bt) for bt in block_tables if bt]
        block_list = []
        block_scales = []
        for i, bt in enumerate(block_tables):
            block_list.extend(bt)
            blocks_in_group = len(bt)
            if blocks_in_group > 0:
                scale = 1.0 / blocks_in_group
                block_scales.extend([scale] * blocks_in_group)

        block_mapping_nested: List[List[int]] = [
            [i] * b_u for i, b_u in enumerate(blocks_used)
        ]
        block_mapping: List[int] = list(
            itertools.chain.from_iterable(block_mapping_nested))

        last_block = [
            sl % block_size + 1 for sl in slot_mapping
        ]
        block_usage = [[block_size] * (b_u - 1) + [lb]
                       for b_u, lb in zip(blocks_used, last_block)]
        block_usage = list(itertools.chain(*block_usage))

        # block_bucket_size = find_bucket(
        #     len(block_list),
        #     self.bucketing_global_state.decode_block_bucket_cfg)
        block_list = pad_list(block_list, bucket_size, _PAD_BLOCK_ID)
        block_mapping = pad_list(block_mapping, bucket_size, -1)
        block_usage = pad_list(block_usage, bucket_size, 1)
        block_scales = pad_list(block_scales, bucket_size, 0.0)

        block_list = torch.tensor(block_list,
                                  dtype=torch.int,
                                  device=self.device)
        block_mapping = torch.tensor(block_mapping,
                                     dtype=torch.long,
                                     device=self.device)
        block_usage = torch.tensor(block_usage,
                                   dtype=torch.bfloat16,
                                   device=self.device)

        block_scales = torch.tensor(block_scales,
                                    dtype=torch.bfloat16,
                                    device=self.device)

        return block_list, block_mapping, block_usage, block_scales

    def create_model_input_from_batch_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        device: str = "hpu:0"
    ) -> ModelInputForHPUWithSamplingMetadata:
        """Convert ModelWorkerBatch to ModelInputForHPUWithSamplingMetadata."""

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=128
        )
        
        batch_size = len(model_worker_batch.req_pool_indices)
        buketized_len = 128
        
        # Initialize input tensors for each sequence
        input_tokens = []
        input_positions = []
        slot_mappings = []
        
        # Track sequence info
        seq_lens = []
        query_lens = []
        
        # Create tensors for each sequence in batch
        for i in range(batch_size):
            # Create padded input token tensor
            input_token = torch.zeros(buketized_len, dtype=torch.int32, device=device)
            seq_len = model_worker_batch.extend_seq_lens[i]
            start_idx = sum(model_worker_batch.extend_seq_lens[:i])
            input_token[:seq_len] = model_worker_batch.input_ids[start_idx:start_idx+seq_len]
            
            input_tokens.append(input_token)
            
            # Create position tensor
            position = torch.zeros(buketized_len, dtype=torch.int32, device=device)
            position[:seq_len] = torch.arange(seq_len, device=device)
            input_positions.append(position)
            
            # Create slot mapping tensor
            slot_mapping = torch.zeros(buketized_len, dtype=torch.int32, device=device)
            slot_mapping[:seq_len] = model_worker_batch.out_cache_loc[start_idx:start_idx+seq_len]
            slot_mappings.append(slot_mapping)
            seq_lens.append(seq_len)
            query_lens.append(seq_len)

        # Stack tensors
        input_tokens = torch.stack(input_tokens)
        input_positions = torch.stack(input_positions)
        slot_mappings = torch.stack(slot_mappings)

        block_indices, block_offsets = precompute_indices_and_offsets(
            self.vllm_config.cache_config.block_size,
            slot_mappings,
            is_prompt=model_worker_batch.forward_mode.is_extend()
        )

        # Create attention metadata
        attn_metadata = HPUAttentionMetadata(
            num_prefills=batch_size,
            num_prefill_tokens=sum(seq_lens),
            num_decode_tokens=0,
            slot_mapping=slot_mappings,
            multi_modal_placeholder_index_maps=None,
            block_list=None,
            block_mapping=None,
            block_usage=None,
            block_indices=block_indices,
            block_offsets=block_offsets,
            block_scales=None,
            is_prompt=model_worker_batch.forward_mode.is_extend(),
            attn_bias=None,
            seq_lens_tensor=torch.tensor(seq_lens, device=device)
        )

        seq_groups = []
        for i in range(batch_size):
            seq_groups.append(SequenceGroupToSample(
                seq_ids=[model_worker_batch.req_pool_indices[i]],
                seq_data={model_worker_batch.req_pool_indices[i]: SequenceData(
                    _prompt_token_ids=array('l', input_tokens[i].tolist())
                )},
                sampling_params=sampling_params,
                seq_len=seq_lens[i],
                query_len=query_lens[i],
                generator=None,
                is_prompt=True,
                prompt_logprob_indices=[],
                sample_indices=[i]
            ))
        
        # selected_token_indices is the last token index of each sequence when flattening the batch after padding
        selected_token_indices = torch.tensor([seq_lens[i] - 1 + i*buketized_len for i in range(batch_size)], device=device)

        # Create sampling metadata
        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,  # Will be populated from batch info
            selected_token_indices=selected_token_indices,
            categorized_sample_indices={
                SamplingType.GREEDY: torch.tensor([], device=device, dtype=torch.int32),
                SamplingType.RANDOM: torch.arange(batch_size, device=device, dtype=torch.int32),
                SamplingType.RANDOM_SEED: torch.tensor([], device=device, dtype=torch.int32)
            },
            num_prompts=batch_size if model_worker_batch.forward_mode.is_extend() else 0,
            skip_sampler_cpu_output=False,
            reuse_sampling_tensors=False
        )

        model_input = ModelInputForHPUWithSamplingMetadata(
            input_tokens=input_tokens,
            input_positions=input_positions,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_mapping=None,
            lora_requests=set(),
            attn_metadata=attn_metadata,
            multi_modal_kwargs={},
            real_batch_size=batch_size,
            batch_size_padded=batch_size,
            virtual_engine=0,
            lora_ids=[0] * batch_size,
            async_callback=None,
            sampling_metadata=sampling_metadata
        )

        return model_input

def create_model_input_prompt() -> ModelInputForHPUWithSamplingMetadata:
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=128
    )
    input_token_s1 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_token_s1[:6] = torch.tensor([128000, 9906, 11, 856, 836, 374], device='hpu:0')
    input_token_s2 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_token_s2[:8] = torch.tensor([128000, 791, 4872, 315, 279, 3723, 4273, 374], device='hpu:0')
    input_token_s3 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_token_s3[:6] = torch.tensor([128000, 791, 6864, 315, 9822, 374], device='hpu:0')
    input_token_s4 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_token_s4[:6] = torch.tensor([128000, 791, 3938, 315, 15592, 374], device='hpu:0')

    input_positions_1 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_positions_1[:6] = torch.tensor([0, 1, 2, 3, 4, 5], device='hpu:0')
    input_positions_2 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_positions_2[:8] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device='hpu:0')
    input_positions_3 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_positions_3[:6] = torch.tensor([0, 1, 2, 3, 4, 5], device='hpu:0')
    input_positions_4 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    input_positions_4[:6] = torch.tensor([0, 1, 2, 3, 4, 5], device='hpu:0')

    slot_mapping_1 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    slot_mapping_1[:6] = torch.tensor([128, 129, 130, 131, 132, 133], device='hpu:0')
    slot_mapping_2 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    slot_mapping_2[:8] = torch.tensor([256, 257, 258, 259, 260, 261, 262, 263], device='hpu:0')
    slot_mapping_3 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    slot_mapping_3[:6] = torch.tensor([384, 385, 386, 387, 388, 389], device='hpu:0')
    slot_mapping_4 = torch.zeros(128, dtype=torch.int64, device='hpu:0')
    slot_mapping_4[:6] = torch.tensor([512, 513, 514, 515, 516, 517], device='hpu:0')

    attn_metadata = HPUAttentionMetadata(
            num_prefills=4, 
            num_prefill_tokens=26, 
            num_decode_tokens=0, 
            slot_mapping=torch.stack([slot_mapping_1, slot_mapping_2, slot_mapping_3, slot_mapping_4]),
            multi_modal_placeholder_index_maps=None,
            block_list=None,
            block_mapping=None,
           block_usage=None,
           block_indices=torch.tensor([1, 2, 3, 4], device='hpu:0'),
           block_offsets=None,
           block_scales=None,
           is_prompt=True, 
           attn_bias=None, 
           seq_lens_tensor=torch.tensor([6, 8, 6, 6], device='hpu:0')
           )

    model_input = ModelInputForHPUWithSamplingMetadata(
        input_tokens=torch.stack([input_token_s1, input_token_s2, input_token_s3, input_token_s4]),    
        input_positions=torch.stack([input_positions_1, input_positions_2, input_positions_3, input_positions_4]),
        seq_lens=[6, 8, 6, 6],
        query_lens=[6, 8, 6, 6],
        lora_mapping=None,
        lora_requests=set(),
        attn_metadata=attn_metadata,
        multi_modal_kwargs={}, 
        real_batch_size=4, 
        batch_size_padded=4, 
        virtual_engine=0, 
        lora_ids=[0, 0, 0, 0], 
        async_callback=None,
        sampling_metadata=SamplingMetadata(
            seq_groups=[SequenceGroupToSample(seq_ids=[0], 
                                             seq_data={0: SequenceData(_prompt_token_ids=array('l', [128000, 9906, 11, 856, 836, 374]))}, 
                                            sampling_params=sampling_params,
                                            seq_len=6,
                                            query_len=6,
                                            generator=None,
                                            is_prompt=True,
                                            prompt_logprob_indices=[],
                                            sample_indices=[0]
                                            ),
                        SequenceGroupToSample(seq_ids=[1], 
                                                seq_data={1: SequenceData(_prompt_token_ids=array('l', [128000, 791, 4872, 315, 279, 3723, 4273, 374]))}, 
                                                sampling_params=sampling_params,
                                                seq_len=8,
                                                query_len=8,
                                                generator=None,
                                                is_prompt=True,
                                                prompt_logprob_indices=[],
                                                sample_indices=[1]
                                                ),
                        SequenceGroupToSample(seq_ids=[2], 
                                                seq_data={2: SequenceData(_prompt_token_ids=array('l', [128000, 791, 6864, 315, 9822, 374]))}, 
                                                sampling_params=sampling_params,
                                                seq_len=6,
                                                query_len=6,
                                                generator=None,
                                                is_prompt=True,
                                                prompt_logprob_indices=[],
                                                sample_indices=[2]
                                                ),
                        SequenceGroupToSample(seq_ids=[3], 
                                                seq_data={3: SequenceData(_prompt_token_ids=array('l', [128000, 791, 3938, 315, 15592, 374]))}, 
                                                sampling_params=sampling_params,
                                                seq_len=6,
                                                query_len=6,
                                                generator=None,
                                                is_prompt=True,
                                                prompt_logprob_indices=[],
                                                sample_indices=[3]
                                                )],
            selected_token_indices=torch.tensor([5, 135, 261, 389], device='hpu:0'),
            categorized_sample_indices={
                SamplingType.GREEDY: torch.tensor([], device='hpu:0', dtype=torch.int32),
                SamplingType.RANDOM: torch.tensor([0, 1, 2, 3], device='hpu:0', dtype=torch.int32),
                SamplingType.RANDOM_SEED: torch.tensor([], device='hpu:0', dtype=torch.int32)
            },
            num_prompts=4,
            skip_sampler_cpu_output=False,
            reuse_sampling_tensors=False
        )
    )
    return model_input

def create_engine_args(
    model: str,
    tokenizer: Optional[str] = None,
    tokenizer_mode: str = "auto",
    skip_tokenizer_init: bool = False,
    trust_remote_code: bool = False,
    allowed_local_media_path: str = "",
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    quantization: Optional[str] = None,
    revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    seed: int = 0,
    gpu_memory_utilization: float = 0.9,
    swap_space: float = 4,
    cpu_offload_gb: float = 0,
    enforce_eager: Optional[bool] = None,
    max_seq_len_to_capture: int = 8192,
    disable_custom_all_reduce: bool = False,
    disable_async_output_proc: bool = False,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    # After positional args are removed, move this right below `model`
    task = "auto",
    pooling_type: Optional[str] = None,
    pooling_norm: Optional[bool] = None,
    pooling_softmax: Optional[bool] = None,
    pooling_step_tag_id: Optional[int] = None,
    pooling_returned_token_ids: Optional[List[int]] = None,
    **kwargs,
) -> None:
    '''
    LLM constructor.

    Note: if enforce_eager is unset (enforce_eager is None)
    it defaults to False.
    '''

    if "disable_log_stats" not in kwargs:
        kwargs["disable_log_stats"] = True

    engine_args = EngineArgs(
        model=model,
        task=task,
        tokenizer=tokenizer,
        tokenizer_mode=tokenizer_mode,
        skip_tokenizer_init=skip_tokenizer_init,
        trust_remote_code=trust_remote_code,
        allowed_local_media_path=allowed_local_media_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        quantization=quantization,
        revision=revision,
        tokenizer_revision=tokenizer_revision,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=swap_space,
        cpu_offload_gb=cpu_offload_gb,
        enforce_eager=enforce_eager,
        max_seq_len_to_capture=max_seq_len_to_capture,
        disable_custom_all_reduce=disable_custom_all_reduce,
        disable_async_output_proc=disable_async_output_proc,
        mm_processor_kwargs=mm_processor_kwargs,
        pooling_type=pooling_type,
        pooling_norm=pooling_norm,
        pooling_softmax=pooling_softmax,
        pooling_step_tag_id=pooling_step_tag_id,
        pooling_returned_token_ids=pooling_returned_token_ids,
        **kwargs,
    )
    return engine_args


def create_model_input_decode() -> ModelInputForHPUWithSamplingMetadata:
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=128
    )

    input_tokens = torch.tensor([[386], [4423], [1101], [1618]], device='hpu:0')
    input_positions = torch.tensor([[6], [8], [6], [6]], device='hpu:0')
    seq_lens = []
    query_lens = []
    lora_mapping = None
    lora_requests = set()

    block_list = torch.zeros(128, dtype=torch.int32, device='hpu:0')
    block_list[:4] = torch.tensor([1, 2, 3, 4,], device='hpu:0')
    block_mapping = torch.zeros(128, dtype=torch.int32, device='hpu:0') - 1
    block_mapping[:4] = torch.tensor([0, 1, 2, 3], device='hpu:0')
    block_usage = torch.zeros(128, dtype=torch.bfloat16, device='hpu:0') + 1
    block_usage[:4] = torch.tensor([7., 9., 7., 7.], device='hpu:0')
    block_indices = torch.tensor([1, 2, 3, 4], device='hpu:0')
    block_offsets = torch.tensor([6, 8, 6, 6], device='hpu:0')
    block_scales = torch.zeros(128, dtype=torch.bfloat16, device='hpu:0') + 1
    block_scales[:4] = torch.tensor([1., 1., 1., 1.], device='hpu:0')

    attn_metadata = HPUAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=30,
        slot_mapping=torch.tensor([[134], [264], [390], [518]], device='hpu:0'),
        multi_modal_placeholder_index_maps=None,
        block_list=block_list,
        block_mapping=block_mapping,
        block_usage=block_usage,
        block_indices=block_indices,
        block_offsets=block_offsets,
        block_scales=block_scales,
        is_prompt=False,
        attn_bias=None,
        seq_lens_tensor=None,
    )

    model_input = ModelInputForHPUWithSamplingMetadata(
        input_tokens=input_tokens,
        input_positions=input_positions,
        seq_lens=seq_lens,
        query_lens=query_lens,
        lora_mapping=lora_mapping,
        lora_requests=lora_requests,
        attn_metadata=attn_metadata,
        multi_modal_kwargs={},
        real_batch_size=4,
        batch_size_padded=4,
        virtual_engine=0,
        lora_ids=[0, 0, 0, 0],
        async_callback=None,
        sampling_metadata=SamplingMetadata(
            seq_groups=[
                SequenceGroupToSample(seq_ids=[0],
                                      sampling_params=sampling_params,
                                      seq_data={0: SequenceData(_prompt_token_ids=array('l', [128000, 9906, 11, 856, 836, 374]),
                                                                                        _output_token_ids=array('l', [386]),
                                                                                        _num_computed_tokens=6)},
                                      seq_len=None,
                                      query_len=1,
                                      generator=None,
                                      is_prompt=False,
                                      prompt_logprob_indices=[],
                                      sample_indices=[0]
                                      ),
                SequenceGroupToSample(seq_ids=[1],
                                      sampling_params=sampling_params,
                                      seq_data={1: SequenceData(_prompt_token_ids=array('l', [128000, 791, 4872, 315, 279, 3723, 4273, 374]),
                                                                                        _output_token_ids=array('l', [4423]),
                                                                                        _num_computed_tokens=8)},
                                      seq_len=None,
                                      query_len=1,
                                      generator=None,
                                      is_prompt=False,
                                      prompt_logprob_indices=[],
                                      sample_indices=[1]
                                      ),
                SequenceGroupToSample(seq_ids=[2],
                                      sampling_params=sampling_params,
                                      seq_data={2: SequenceData(_prompt_token_ids=array('l', [128000, 791, 6864, 315, 9822, 374]),
                                                                                        _output_token_ids=array('l', [1101]),
                                                                                        _num_computed_tokens=6)},
                                      seq_len=None,
                                      query_len=1,
                                      generator=None,
                                      is_prompt=False,
                                      prompt_logprob_indices=[],
                                      sample_indices=[2]
                                      ),
                SequenceGroupToSample(seq_ids=[3],
                                      sampling_params=sampling_params,
                                      seq_data={3: SequenceData(_prompt_token_ids=array('l', [128000, 791, 3938, 315, 15592, 374]),
                                                                                        _output_token_ids=array('l', [1618]),
                                                                                        _num_computed_tokens=6)},
                                      seq_len=None,
                                      query_len=1,
                                      generator=None,
                                      is_prompt=False,
                                      prompt_logprob_indices=[],
                                      sample_indices=[3]
                                      ),
            ],
            selected_token_indices=torch.tensor([0, 1, 2, 3], device='hpu:0'),
            categorized_sample_indices={
                SamplingType.GREEDY: torch.tensor([], device='hpu:0', dtype=torch.int32),
                SamplingType.RANDOM: torch.tensor([0, 1, 2, 3], device='hpu:0', dtype=torch.int32),
                SamplingType.RANDOM_SEED: torch.tensor([], device='hpu:0', dtype=torch.int32)
            },
            num_prompts=0,
            skip_sampler_cpu_output=False,
            reuse_sampling_tensors=False
        )
    )
    return model_input


def main():
    
    engine_args = create_engine_args(
        model="meta-llama/Llama-3.2-1B", device="hpu", dtype="bfloat16"
    )
    vllm_config = engine_args.create_engine_config()

    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port())

    # Initialize HPU worker
    worker = HPUWorker(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method=distributed_init_method,
        is_driver_worker=True
    )

    # # Initialize device and load model
    worker.init_device()
    worker.load_model()

    # Sample prompts
    prompt = "Hello, my name is"

    # Create sampling parameters

    num_gpu_blocks, num_cpu_blocks = worker.determine_num_available_blocks()
    worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    worker_input = WorkerInput(
        num_seq_groups=4,
        blocks_to_swap_in=torch.tensor([], device='hpu:0', dtype=torch.int64).view(-1, 2),
        blocks_to_swap_out=torch.tensor([], device='hpu:0', dtype=torch.int64).view(-1, 2),
        blocks_to_copy=torch.tensor([], device='hpu:0', dtype=torch.int64).view(-1, 2),
        virtual_engine=0,
        num_steps=1
    )

    # worker.execute_worker(worker_input)

    model_input = create_model_input_prompt()

    output = worker.model_runner.execute_model(
        model_input=model_input,
        kv_caches=worker.kv_cache[worker_input.virtual_engine] if worker.kv_cache is not None else None,
        intermediate_tensors=None,
        num_steps=worker_input.num_steps,
        **{}
    )

    print(output)

    model_input = create_model_input_decode()

    output = worker.model_runner.execute_model(
        model_input=model_input,
        kv_caches=worker.kv_cache[worker_input.virtual_engine] if worker.kv_cache is not None else None,
        intermediate_tensors=None,
        num_steps=worker_input.num_steps,
        **{}
    )

    print(output)

    # outputs = worker.execute_model(
    #     execute_model_req=worker._prepare_worker_input(seq_group_metadata_list)
    # )

    # print(outputs)

if __name__ == "__main__":
    main() 