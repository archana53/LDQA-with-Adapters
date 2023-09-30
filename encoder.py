import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from enum import Enum
from dataclasses import dataclass


@dataclass
class LongformerEncoder:
    model_id = "allenai/longformer-base-4096"
    tokenizer_id = "allenai/longformer-base-4096"

    def get_global_attention_mask(self, input_ids, frequency=16):
        """Create a global attention mask to attend to the first token every `frequency` tokens."""
        attention_mask = torch.zeros_like(input_ids)
        attention_mask[:, 0::frequency] = 1
        return attention_mask


@dataclass
class LongT5Encoder:
    model_id = "google/long-t5-local-base"
    tokenizer_id = "google/long-t5-local-base"


@dataclass
class LLaMaEncoder:
    model_id = "meta-llama/Llama-2-7b-hf"
    tokenizer_id = "meta-llama/Llama-2-7b-hf"


class EncoderType(Enum):
    """Type of encoder to use"""
    LongFormer = LongformerEncoder
    LongT5 = LongT5Encoder
    LLaMa = LLaMaEncoder


class Encoder:
    def __init__(
        self, encoder: EncoderType, chunk_size: int = 4096, pooled_outputs: bool = False
    ):
        """
        :param encoder: Type of encoder to use
        :param chunk_size: Size of chunks to split the document into
        :param pooled_outputs: Whether to return pooled outputs
        """
        self.chunk_size = chunk_size
        self.pooled_outputs = pooled_outputs

        if encoder not in EncoderType:
            raise ValueError(f"Encoder must be one of {list(EncoderType)}")
        
        self.encoder_type = encoder
        self.encoder = EncoderType(encoder)
        self.model = AutoModel.from_pretrained(self.encoder.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder.tokenizer_id)

        if self.model.config.max_position_embeddings < self.chunk_size:
            raise ValueError(
                f"Chunk size is greater than maximum sequence length for {encoder} encoder"
            )

        self.tokenizer.model_max_length = self.model.config.max_position_embeddings

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a long document using an encoder
        :param text: Document to encode
        :return: Encoded document
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt").unsqueeze(0)

        # encode each chunk
        outputs = []
        for i in range(0, input_ids.shape[1], self.chunk_size):
            chunk_input_ids = input_ids[:, i : i + self.chunk_size]
            if self.encoder_type == EncoderType.LongFormer:
                global_attention_mask = self.encoder.get_global_attention_mask(chunk_input_ids)
                output = self.model(
                    input_ids=chunk_input_ids, global_attention_mask=global_attention_mask
                )
            else:
                output = self.model(input_ids=chunk_input_ids)
            outputs.append(output)

        # concatenate outputs
        if self.pooled_outputs:
            return torch.cat([output.pooler_output for output in outputs], dim=1)
        else:
            return torch.cat([output.last_hidden_state for output in outputs], dim=1)
