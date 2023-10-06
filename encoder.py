from dataclasses import dataclass
from enum import Enum

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class EncoderConfig:
    model_id: str = None
    tokenizer_id: str = None

    def get_model(self):
        return AutoModel.from_pretrained(self.model_id)

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer_id)


@dataclass
class LongformerEncoderConfig(EncoderConfig):
    model_id: str = "allenai/longformer-base-4096"
    tokenizer_id: str = "allenai/longformer-base-4096"

    @staticmethod
    def get_global_attention_mask(input_ids, frequency=16):
        """Create a global attention mask to attend to the first token every `frequency` tokens."""
        attention_mask = torch.zeros_like(input_ids)
        attention_mask[:, 0::frequency] = 1
        return attention_mask


@dataclass
class LongT5EncoderConfig(EncoderConfig):
    model_id: str = "google/long-t5-local-base"
    tokenizer_id: str = "google/long-t5-local-base"


@dataclass
class LLaMaEncoderConfig(EncoderConfig):
    model_id: str = "meta-llama/Llama-2-7b-hf"
    tokenizer_id: str = "meta-llama/Llama-2-7b-hf"


class EncoderType(Enum):
    """Type of encoder to use"""

    LongFormer = LongformerEncoderConfig
    LongT5 = LongT5EncoderConfig
    LLaMa = LLaMaEncoderConfig


if __name__ == "__main__":
    # try to instantiate the encoder
    encoder = EncoderType.LongFormer.value()
    print(encoder.get_model())
