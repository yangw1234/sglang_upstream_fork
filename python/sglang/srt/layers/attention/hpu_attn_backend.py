from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn.functional import scaled_dot_product_attention
import vllm_hpu_extension.kernels as kernels

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
import vllm_hpu_extension.ops as ops
from vllm_hpu_extension.utils import (Matmul, ModuleFusedSDPA, Softmax,
                                      VLLMKVCache)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class HPUAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.k_cache = VLLMKVCache()
        self.v_cache = VLLMKVCache()
        from habana_frameworks.torch.hpex.kernels import FusedSDPA
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(
                    FusedSDPA)
        self.matmul_qk = Matmul()
        self.softmax = Softmax()
        self.matmul_av = Matmul()
        self.batch2block_matmul = Matmul()
        self.block2batch_matmul = Matmul()


    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        import vllm_hpu_extension.environment as environment
        # TODO: remove the hardcoded model_type once we have a better way to handle this
        environment.runtime_params['model_type'] = 'llama'


    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.block_indices, forward_batch.block_offsets, k, v
            )

        query = q.view(1, -1, layer.tp_q_head_num, layer.qk_head_dim)
        key = k.view(1, -1, layer.tp_k_head_num, layer.qk_head_dim)
        value = v.view(1, -1, layer.tp_v_head_num, layer.v_head_dim)

        output = ops.prompt_attention(
            query,
            key,
            value,
            attn_bias=forward_batch.attn_bias,
            p=0.0,
            scale=layer.scaling,
            matmul_qk_op=self.matmul_qk,
            softmax_op=self.softmax,
            matmul_av_op=self.matmul_av,
            valid_seq_lengths=forward_batch.valid_seq_len,
            fsdpa_op=self.fused_scaled_dot_product_attention,
        )
        output = output.reshape(q.shape)

        return output

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.block_indices, forward_batch.block_offsets, k, v
            )


        # Get key and value caches
        key_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        value_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        
        query = q.view(-1, 1, layer.tp_q_head_num * layer.qk_head_dim)
        # key_cache = key_cache.view(-1, forward_batch.page_size, layer.tp_k_head_num, layer.qk_head_dim)
        # value_cache = value_cache.view(-1, forward_batch.page_size, layer.tp_v_head_num, layer.v_head_dim)

        # if save_kv_cache:
        #     key = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        #     value = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        #     key_cache.index_put_((block_indices, block_offsets), key)
        #     value_cache.index_put_((block_indices, block_offsets), value)

        # Run paged attention decode
        output = ops.flat_pa(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_list=forward_batch.block_list,
            block_mapping=forward_batch.block_mapping,
            block_bias=forward_batch.attn_bias,
            block_scales=forward_batch.block_scales,
            block_groups=forward_batch.block_groups,
            scale=layer.scaling,
            matmul_qk_op=self.matmul_qk,
            matmul_av_op=self.matmul_av,
            batch2block_matmul_op=self.batch2block_matmul,
            block2batch_matmul_op=self.block2batch_matmul,
            keys_fetch_func=self.k_cache.fetch_from_cache,
            values_fetch_func=self.v_cache.fetch_from_cache,
        )

        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
