# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

import enum
import logging
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
import torch.nn.functional as F
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton import (
        FusedMoE,
        FusedMoEMethodBase,
        FusedMoeWeightScaleSupported,
    )

from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.layers.quantization.utils import (
    all_close_1d,
    is_cuda,
    is_fp8_fnuz,
    per_tensor_dequantize,
    replace_parameter,
)
from sglang.srt.utils import set_weight_attrs, is_hpu
from sglang.srt.custom_op import CustomOp
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_is_cuda = is_cuda()
_is_hpu = is_hpu()

if _is_cuda:
    from sglang.srt.custom_op import scaled_fp8_quant as sgl_scaled_fp8_quant
else:
    from vllm import _custom_ops as vllm_ops

try:
    import vllm

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

if _is_hpu:
    import habana_frameworks.torch.utils.experimental as htexp
    if htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi2:
        FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max
    import habana_frameworks.torch as htorch
    from vllm_hpu_extension.ops import scaled_fp8_quant
    vllm_ops.scaled_fp8_quant = scaled_fp8_quant

logger = logging.getLogger(__name__)


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


__all__ = [
    "CompressedTensorsMoEMethod",
    "CompressedTensorsW8A8Fp8MoEMethod",
    "CompressedTensorsWNA16MoEMethod",
]


class CompressedTensorsMoEMethod:
    def __new__(cls, *args, **kwargs):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoEMethodBase

        if cls is CompressedTensorsMoEMethod:
            return super().__new__(cls)
        return super().__new__(cls)

    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
    ) -> "CompressedTensorsMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get("input_activations")

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            if not VLLM_AVAILABLE:
                raise ImportError(
                    "vllm is not installed, to use CompressedTensorsWNA16MoEMethod, please install vllm"
                )
            return CompressedTensorsWNA16MoEMethod(quant_config)
        elif quant_config._is_fp8_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Fp8MoEMethod(quant_config)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )


class CompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsMoEMethod):

    def __init__(
        self, quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoEMethodBase,
            FusedMoeWeightScaleSupported,
        )

        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations"
        )

        self.static_input_scales = not self.input_quant.dynamic

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # per-tensor quantization
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            weight_quant_method = FusedMoeWeightScaleSupported.TENSOR.value
        elif self.weight_quant.strategy == QuantizationStrategy.CHANNEL:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
                requires_grad=False,
            )
            weight_quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
        else:
            raise ValueError(
                f"Unsupported weight quantization strategy: {self.weight_quant.strategy}"
            )

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update({"quant_method": weight_quant_method})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            assert (
                self.input_quant.strategy == QuantizationStrategy.TENSOR
            ), "Only per-tensor quantization is supported for static input scales"
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.static_input_scales:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                layer.w2_input_scale
            ):
                logger.warning(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer."
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        if not _is_hpu and is_fp8_fnuz():
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = (
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                )
            )
            w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
            )
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(
                    w13_input_scale, requires_grad=False
                )
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(
                    w2_input_scale, requires_grad=False
                )
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            print("***************************** got here ******************")
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )

                    if _is_cuda:
                        (
                            layer.w13_weight[expert_id][start : start + shard_size, :],
                            _,
                        ) = sgl_scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                    else:
                        (
                            layer.w13_weight[expert_id][start : start + shard_size, :],
                            _,
                        ) = vllm_ops.scaled_fp8_quant(
                            dq_weight, max_w13_scales[expert_id]
                        )
                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )
    
    def apply(self, *args, **kwargs):
        if _is_hpu:
            return self.forward_hpu(*args, **kwargs)
        else:
            return self.forward_cuda(*args, **kwargs)

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        inplace: bool = True,
        no_combine: bool = False,
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton import fused_experts
        from sglang.srt.layers.moe.topk import select_experts

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
        )

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            activation=activation,
            use_fp8_w8a8=True,
            per_channel_quant=self.weight_quant.strategy
            == QuantizationStrategy.CHANNEL,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

    def forward_hpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        inplace: bool = True,
        no_combine: bool = False,
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton import fused_experts
        from sglang.srt.layers.moe.topk import select_experts
        import habana_frameworks.torch as htorch
        hidden_dim = x.shape[-1]
        num_experts = layer.w13_weight.shape[0]
        moe_n_slice = 8 if num_experts > 32 else 1
        n_expert_slice = num_experts // moe_n_slice
        assert n_expert_slice * moe_n_slice == num_experts
        x = x.view(-1, hidden_dim)

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
        )
        topk_weights = topk_weights.view(-1, top_k)
        topk_ids = topk_ids.view(-1, top_k)
        ep_rank = 0
        assert ep_rank is not None
        ep_shift = ep_rank * num_experts

        def naive_dequant_moe(x, topk_ids, topk_weights, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale):

            w13_weight = w13_weight.to(torch.bfloat16) * w13_weight_scale.to(torch.bfloat16)
            w2_weight = w2_weight.to(torch.bfloat16) * w2_weight_scale.to(torch.bfloat16)

            for i in range(moe_n_slice):
                min_expert = i * n_expert_slice
                max_expert = (i + 1) * n_expert_slice
                w13_list_slice = [
                    w13_weight[j]
                    for j in range(min_expert, max_expert)
                ]
                w2_list_slice = [
                    w2_weight[j]
                    for j in range(min_expert, max_expert)
                ]
                current_hidden_states = torch.ops.hpu.mixture_of_experts(
                    hidden_states=x,
                    expert_routing_table=topk_ids.to(torch.int64),
                    router_weights=topk_weights.to(x.dtype),
                    w12=w13_list_slice,
                    w3=w2_list_slice,
                    permuted_weights=True,
                    activation=activation,
                    experts_min=min_expert + ep_shift,
                    experts_max=max_expert - 1 + ep_shift,)
                htorch.core.mark_step()
                torch.hpu.synchronize()
                if i == 0:
                    final_hidden_states = current_hidden_states
                else:
                    final_hidden_states.add_(current_hidden_states)
            return final_hidden_states.view(-1, x.shape[1])   



        def dynamic_quant(data, single_scale = False):
            if single_scale:
                scale = ((torch.abs(data)).max() + 1e-8) / FP8_MAX
            else:
                scale = ((torch.abs(data)).max(dim=-1).values + 1e-8) / FP8_MAX
                scale = scale.unsqueeze(-1)
            data_fp8 = torch.ops.hpu.cast_to_fp8_v2(data, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
            return data_fp8, scale.float()

        def do_static_moe_with_dynamic_scaling(x, topk_ids, topk_weights, w13_weight_fp8, w2_weight_fp8, total_num_experts, num_experts, w13_weight_scale_inv_fp8=None, w2_weight_scale_inv_fp8=None):
            router_indices = topk_ids.reshape(-1, 1).expand(-1, hidden_dim)
            print(f"router_indices: {router_indices.size()} {router_indices.dtype} {router_indices.device}")
            routed_in = torch.gather(
                input=x,
                dim=0,
                index=router_indices,
            )
            print(f"routed_in: {routed_in.size()} {routed_in.dtype} {routed_in.device}")
            routed_in = routed_in * topk_weights.reshape(-1, 1)
            print(f"routed_in: {routed_in.size()} {routed_in.dtype} {routed_in.device}")
            routed_in = routed_in.view(num_experts, -1, hidden_dim)
            print(f"routed_in: {routed_in.size()} {routed_in.dtype} {routed_in.device}")
            x_fp8, x_scale = dynamic_quant(routed_in)
            print(f"x_fp8: {x_fp8.size()} {x_fp8.dtype} {x_fp8.device}")
            print(f"x_scale: {x_scale.size()} {x_scale.dtype} {x_scale.device}")

            gate_up = torch.ops.hpu.fp8_gemm_v2(
                    A=x_fp8,
                    trans_A=False,
                    B=w13_weight_fp8,
                    trans_B=True,
                    D=None,
                    out_dtype=torch.bfloat16,
                    A_scale_inv=x_scale,
                    B_scale_inv=w13_weight_scale_inv_fp8.transpose(1, 2),
                    bias=None,
                    accumulate=False)
            gate, up = gate_up.chunk(2, dim=-1)
            gate_up = F.silu(up) * gate
            gate_up, gate_up_scale = dynamic_quant(gate_up)
            current_hidden_states = torch.ops.hpu.fp8_gemm_v2(
                gate_up,
                False,
                w2_weight_fp8,
                True,
                None,
                torch.bfloat16,
                gate_up_scale,
                w2_weight_scale_inv_fp8.transpose(1, 2),
                None,
                False,
            )
            base = torch.zeros_like(x)
            result = base.scatter_add(dim=0, index=router_indices, src=current_hidden_states.view(-1, hidden_dim))
            # print(f"result: {result.size()} {result.dtype} {result.device}")
            return result

        # print("Tensors passed to fused_experts:")
        # print(f"  x: {x.size()} {x.dtype} {x.device}")
        # print(f"  layer.w13_weight: {layer.w13_weight.size()} {layer.w13_weight.dtype} {layer.w13_weight.device}")
        # print(f"  layer.w2_weight: {layer.w2_weight.size()} {layer.w2_weight.dtype} {layer.w2_weight.device}")
        # print(f"  topk_weights: {topk_weights.size()} {topk_weights.dtype} {topk_weights.device}")
        # print(f"  topk_ids: {topk_ids.size()} {topk_ids.dtype} {topk_ids.device}")
        # print(f"  inplace: {inplace}")
        # print(f"  activation: {activation}")
        # print(f"  use_fp8_w8a8: {True}")
        # print(f"  per_channel_quant: {self.weight_quant.strategy == QuantizationStrategy.CHANNEL}")
        # print(f"  w1_scale: {layer.w13_weight_scale.size()} {layer.w13_weight_scale.dtype} {layer.w13_weight_scale.device}")
        # print(f"  w2_scale: {layer.w2_weight_scale.size()} {layer.w2_weight_scale.dtype} {layer.w2_weight_scale.device}")
        # print(f"  a1_scale: {layer.w13_input_scale.size() if layer.w13_input_scale is not None else 'None'} {layer.w13_input_scale.dtype if layer.w13_input_scale is not None else 'None'} {layer.w13_input_scale.device if layer.w13_input_scale is not None else 'None'}")
        # print(f"  a2_scale: {layer.w2_input_scale.size() if layer.w2_input_scale is not None else 'None'} {layer.w2_input_scale.dtype if layer.w2_input_scale is not None else 'None'} {layer.w2_input_scale.device if layer.w2_input_scale is not None else 'None'}")
        # print(f"  apply_router_weight_on_input: {apply_router_weight_on_input}")


        # result = do_static_moe_with_dynamic_scaling(x,
        #                                            topk_ids,
        #                                            topk_weights,
        #                                            layer.w13_weight,
        #                                            layer.w2_weight,
        #                                            moe_n_slice,
        #                                            n_expert_slice,
        #                                            layer.w13_weight_scale,
        #                                            layer.w2_weight_scale)
        result = naive_dequant_moe(x,
                                   topk_ids,
                                   topk_weights,
                                   layer.w13_weight,
                                   layer.w2_weight,
                                   layer.w13_weight_scale,
                                   layer.w2_weight_scale)
        htorch.core.mark_step()
        return result

class CompressedTensorsWNA16MoEMethod(CompressedTensorsMoEMethod):

    def __init__(
        self, quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoEMethodBase,
            FusedMoeWeightScaleSupported,
        )

        self.quant_config = quant_config
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        config = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.strategy = config.strategy
        self.group_size = config.group_size
        self.actorder = config.actorder
        assert config.symmetric, "Only symmetric quantization is supported for MoE"

        if not (
            self.quant_config.quant_format == CompressionFormat.pack_quantized.value
            and self.num_bits in WNA16_SUPPORTED_BITS
        ):
            raise ValueError(
                "For Fused MoE layers, only ",
                f"{CompressionFormat.pack_quantized.value} ",
                "is supported for the following bits: ",
                f"{WNA16_SUPPORTED_BITS}",
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        assert (
            params_dtype == torch.float16
        ), "float16 is required for MoE compressed models. Set dtype=torch.float16"  # noqa: E501

        intermediate_size_full = extra_weight_attrs.pop("intermediate_size_full")

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update(
            {"is_transposed": True, "quant_method": self.strategy}
        )
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.packed_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.packed_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # In the case where we have actorder/g_idx,
        # we do not partition the w2 scales
        load_full_w2 = self.actorder and self.group_size != -1
        w2_scales_size = (
            intermediate_size_full if load_full_w2 else intermediate_size_per_partition
        )

        self.is_k_full = (not self.actorder) or (
            intermediate_size_per_partition == intermediate_size_full
        )

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None
        layer.marlin_state = GPTQMarlinState.REPACK

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        def replace_tensor(name, new_t):
            # It is important to use resize_() here since it ensures
            # the same buffer is reused
            getattr(layer, name).resize_(new_t.shape)
            getattr(layer, name).copy_(new_t)
            del new_t

        def get_scale_perms(num_bits: int):
            scale_perm: List[int] = []
            for i in range(8):
                scale_perm.extend([i + 8 * j for j in range(8)])
            scale_perm_single: List[int] = []
            for i in range(4):
                scale_perm_single.extend(
                    [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]]
                )
            return scale_perm, scale_perm_single

        def marlin_permute_scales(
            s: torch.Tensor, size_k: int, size_n: int, group_size: int, num_bits: int
        ):
            scale_perm, scale_perm_single = get_scale_perms(num_bits)
            if group_size < size_k and group_size != -1:
                s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
            else:
                s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
            s = s.reshape((-1, size_n)).contiguous()
            return s

        def marlin_moe_permute_scales(
            s: torch.Tensor, size_k: int, size_n: int, group_size: int, num_bits: int
        ):
            num_experts = s.shape[0]
            output = torch.empty(
                (num_experts, s.shape[1], s.shape[2]), device=s.device, dtype=s.dtype
            )
            for e in range(num_experts):
                output[e] = marlin_permute_scales(
                    s[e], size_k, size_n, group_size, num_bits
                )
            return output

        size_k2 = layer.w2_weight_packed.shape[2]
        size_k13 = layer.w13_weight_packed.shape[2]

        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device

        # when running models with grouped act order,
        # resort to g_idx values provided in checkpoint
        if self.actorder == "group":
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)

            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_weight_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_weight_g_idx[e]).to(
                    torch.int32
                )
                w13_sorted_g_idx[e] = layer.w13_weight_g_idx[e][
                    w13_g_idx_sort_indices[e]
                ]
                w2_sorted_g_idx[e] = layer.w2_weight_g_idx[e][w2_g_idx_sort_indices[e]]

            replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)

        else:
            layer.w13_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )

        marlin_w13_qweight = ops.gptq_marlin_moe_repack(
            layer.w13_weight_packed,
            layer.w13_g_idx_sort_indices,
            layer.w13_weight_packed.shape[1] * self.packed_factor,
            layer.w13_weight_packed.shape[2],
            self.num_bits,
        )
        replace_tensor("w13_weight_packed", marlin_w13_qweight)
        marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            layer.w2_weight_packed,
            layer.w2_g_idx_sort_indices,
            layer.w2_weight_packed.shape[1] * self.packed_factor,
            layer.w2_weight_packed.shape[2],
            self.num_bits,
        )
        replace_tensor("w2_weight_packed", marlin_w2_qweight)
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            layer.w13_weight_scale,
            size_k13,
            layer.w13_weight_scale.shape[2],
            self.group_size,
            self.num_bits,
        )
        replace_tensor("w13_weight_scale", marlin_w13_scales)
        marlin_w2_scales = marlin_moe_permute_scales(
            layer.w2_weight_scale,
            layer.w2_weight_scale.shape[1]
            * (self.group_size if self.group_size != -1 else self.packed_factor),
            size_k2,
            self.group_size,
            self.num_bits,
        )
        replace_tensor("w2_weight_scale", marlin_w2_scales)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.moe.topk import select_experts

        assert activation == "silu", "Only SiLU activation is supported."
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vllm is not installed, to use fused_marlin_moe, please install vllm"
            )
        if expert_map is not None:
            raise NotImplementedError(
                "Expert Parallelism is not supported for " "fused Marlin MoE method."
            )

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            correction_bias=correction_bias,
        )

        return torch.ops.vllm.fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            router_logits,
            topk_weights,
            topk_ids,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.num_bits,
            is_k_full=self.is_k_full,
        )
