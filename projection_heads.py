import dataclasses
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch
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
        self.projection = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = x.mean(dim=2)
        x = self.projection(x)
        return x


class MaxPoolProjectionHead(nn.Module):
    """Simple Max pooler head with a linear projection layer."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = x.max(dim=2)
        x = self.projection(x)
        return x


class AttentionProjectionHead(nn.Module):
    """Projection head with multi-head self-attention.
    Receives document chunk embeddings as input.
    Computes self-attention and returns the output of the last layer.

    Call Args:
        x: Document chunk embeddings of shape (batch_size, num_chunks, chunk_size, embedding_dim).
        x_mask: Binary attention mask for document chunk embeddings denoting padding tokens.
                Shape (batch_size, num_chunks, chunk_size).
    """

    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            self.input_dim, self.num_heads, batch_first=True
        )

    def forward(self, x, x_mask=None):
        x = self.attn(x, x, x, need_weights=False, attn_mask=x_mask)[0]
        return x


class QueryAwareProjectionHead(nn.Module):
    """Query Aware Projection head with multi-head self-attention and cross-attention.
    Receives document chunk embeddings and query embeddings as input.
    Computes self-attention on document chunk embeddings first and then cross-attention
    with query embeddings. Returns the output of the last layer.

    Args:
        query_dim: Dimension of query embeddings. Query is the document chunk in this case.
        key_dim: Dimension of key embeddings. Key is the question in this case.
        num_self_attention_heads: Number of self-attention heads for document chunk embeddings.
        num_cross_attention_heads: Number of cross-attention heads for document chunk embeddings.
        outputs_per_chunk: Number of outputs per document chunk.

    Call Args:
        x: Document chunk embeddings of shape (batch_size, num_chunks, chunk_size, embedding_dim).
        x_mask: Binary attention mask for document chunk embeddings denoting padding tokens.
                Shape (batch_size, num_chunks, chunk_size).
        q: Query embeddings of shape (batch_size, query_size, embedding_dim).
        q_mask: Binary attention mask for query embeddings denoting padding tokens.
                Shape (batch_size, query_size).

    Returns:
        x: Query aware document chunk embeddings of shape (batch_size, num_chunks * outputs_per_chunk, embedding_dim).
    """

    def __init__(
        self, query_dim, key_dim, num_self_attention_heads, num_cross_attention_heads, outputs_per_chunk=1
    ):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_self_attention_heads = num_self_attention_heads
        self.num_cross_attention_heads = num_cross_attention_heads
        self.outputs_per_chunk = outputs_per_chunk

        self.self_attn = nn.MultiheadAttention(
            self.query_dim, self.num_self_attention_heads, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            self.query_dim,
            self.num_cross_attention_heads,
            batch_first=True,
            kdim=key_dim,
            vdim=key_dim,
        )

    def forward(self, doc_emb, query_emb, doc_mask=None, query_mask=None):
        # get shapes
        bs, num_chunks, chunk_size, emb_dim = doc_emb.shape
        _, query_size, _ = query_emb.shape
        
        # collapse batch and num_chunks dimensions
        doc_emb = doc_emb.view(bs * num_chunks, chunk_size, emb_dim)
        doc_mask = doc_mask.view(bs * num_chunks, chunk_size)

        # self-attention on document chunks
        x = self.self_attn(doc_emb, doc_emb, doc_emb, need_weights=False, attn_mask=doc_mask)[0]
        x = x.view(bs, num_chunks, chunk_size, emb_dim)

        # cross-attention on document chunks and query
        cross_attn_mask = self.get_attention_mask(doc_mask, query_mask)
        x = x.permute(1, 0, 2, 3)  # shape: (num_chunks, bs, chunk_size, emb_dim)
        cross_attn_mask = cross_attn_mask.permute(1, 0, 2, 3)  # shape: (num_chunks, bs, chunk_size, query_size)

        outputs = []
        for x_chunk, mask_chunk in zip(x, cross_attn_mask):
            x_chunk = self.cross_attn(
                query_emb, x_chunk, x_chunk, need_weights=False, attn_mask=mask_chunk
            )[0]
            outputs.append(x_chunk)
        x = torch.stack(outputs, dim=1)
        x = x.permute(1, 0, 2, 3)  # shape: (bs, num_chunks, chunk_size, emb_dim)

        # get outputs_per_chunk outputs per chunk
        x = x[:, :, :self.outputs_per_chunk, :]
        x = x.view(bs, num_chunks * self.outputs_per_chunk, emb_dim)
        return x

    def get_attention_mask(self, doc_mask, query_mask):
        """Compute attention mask for query aware projection head.
        Args:
            doc_mask: Binary attention mask for document chunk embeddings denoting padding tokens.
                      Shape (batch_size, num_chunks, chunk_size).
            query_mask: Binary attention mask for query embeddings denoting padding tokens.
                        Shape (batch_size, query_size).
        Returns:
            attention_mask: Binary attention mask for query aware projection head.
                            Shape (batch_size, num_chunks, chunk_size, query_size).
        """
        bs, num_chunks, chunk_size = doc_mask.shape
        _, query_size = query_mask.shape

        doc_mask = doc_mask.view(bs, num_chunks, chunk_size, 1)
        query_mask = query_mask.view(bs, 1, 1, query_size)

        attention_mask = doc_mask * query_mask
        return attention_mask
