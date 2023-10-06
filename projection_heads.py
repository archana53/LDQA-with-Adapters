from enum import Enum
from dataclasses import dataclass
from abc import abstractmethod

import dataclasses
import torch.nn as nn


@dataclass
class ProjectionHeadConfig:
    """Configuration for projection head."""

    @abstractmethod
    def get_projection_head(self):
        raise NotImplementedError("Override this method in child class")
    
    @classmethod
    def from_kwargs(cls, **kwargs):
        # fetch the constructor's signature
        field_names = set([f.name for f in dataclasses.fields(cls)])

        # split the kwargs into natives and unknowns
        native_args, new_args = {}, {}
        for name, val in kwargs.items():
            if name in field_names:
                native_args[name] = val
            else:
                new_args[name] = val

        # use the natives to create the object
        ret = cls(**native_args)
        return ret


@dataclass
class AvgPoolProjectionHeadConfig(ProjectionHeadConfig):
    """Configuration for average pool projection head."""

    input_dim: int
    output_dim: int

    def get_projection_head(self):
        return AvgPoolProjectionHead(self.input_dim, self.output_dim)


@dataclass
class MaxPoolProjectionHeadConfig(ProjectionHeadConfig):
    """Configuration for max pool projection head."""

    input_dim: int
    output_dim: int

    def get_projection_head(self):
        return MaxPoolProjectionHead(self.input_dim, self.output_dim)


@dataclass
class AttentionProjectionHeadConfig(ProjectionHeadConfig):
    """Configuration for attention projection head."""

    input_dim: int
    output_dim: int
    num_self_attention_heads: int

    def get_projection_head(self):
        return AttentionProjectionHead(
            self.input_dim, self.output_dim, self.num_self_attention_heads
        )


@dataclass
class QueryAwareProjectionHeadConfig(ProjectionHeadConfig):
    """Configuration for query aware projection head."""

    input_dim: int
    output_dim: int
    num_self_attention_heads: int
    num_cross_attention_heads: int

    def get_projection_head(self):
        return QueryAwareProjectionHead(
            self.input_dim,
            self.output_dim,
            self.num_self_attention_heads,
            self.num_cross_attention_heads,
        )


class ProjectionHeadType(Enum):
    """Enum for projection head types."""

    AvgPool = AvgPoolProjectionHeadConfig
    MaxPool = MaxPoolProjectionHeadConfig
    Attention = AttentionProjectionHeadConfig
    QueryAware = QueryAwareProjectionHeadConfig


class AvgPoolProjectionHead(nn.Module):
    """Simple Average pooler head with a linear projection layer."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.avg_pool(x)
        x = x.squeeze(-1)
        x = self.projection(x)
        return x


class MaxPoolProjectionHead(nn.Module):
    """Simple Max pooler head with a linear projection layer."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.projection = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.max_pool(x)
        x = x.squeeze(-1)
        x = self.projection(x)
        return x


class AttentionProjectionHead(nn.Module):
    """Projection head with multi-head self-attention."""

    # TODO: implement this

    def __init__(self, input_dim, output_dim, num_heads):
        raise NotImplementedError("TODO: implement this")

    def forward(self, x):
        raise NotImplementedError("TODO: implement this")


class QueryAwareProjectionHead(nn.Module):
    """Query Aware Projection head with multi-head self-attention and cross-attention."""

    # TODO: implement this

    def __init__(
        self, input_dim, output_dim, num_self_attention_heads, num_cross_attention_heads
    ):
        raise NotImplementedError("TODO: implement this")

    def forward(self, x, q):
        raise NotImplementedError("TODO: implement this")
