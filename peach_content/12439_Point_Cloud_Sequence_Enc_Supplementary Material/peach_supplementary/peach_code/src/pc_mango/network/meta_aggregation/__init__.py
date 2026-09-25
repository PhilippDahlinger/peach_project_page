import torch

from pc_mango.util.own_types import ConfigDict


def get_meta_aggregation(config: ConfigDict) -> torch.nn.Module:
    aggregation_name = config.name
    match aggregation_name:
        case "max":
            from pc_mango.network.meta_aggregation.max_aggregation import MaxAggregation
            return MaxAggregation(config)
        case "mean":
            from pc_mango.network.meta_aggregation.mean_aggregation import MeanAggregation
            return MeanAggregation(config)
        case "softmax_pool":
            from pc_mango.network.meta_aggregation.softmax_pool import SoftmaxPoolAggregation
            return SoftmaxPoolAggregation(config)
        case "attention":
            from pc_mango.network.meta_aggregation.attention_aggregation import AttentionAggregation
            return AttentionAggregation(config)
        case "gated":
            from pc_mango.network.meta_aggregation.gated_aggregation import GatedAggregation
            return GatedAggregation(config)
        case "set_transformer":
            from pc_mango.network.meta_aggregation.set_transformer_aggregation import SetTransformerAggregation
            return SetTransformerAggregation(config)
        case _:
            raise ValueError(f"Unknown MetaAggregation {aggregation_name}")
