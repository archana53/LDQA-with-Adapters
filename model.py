import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class LDQAModelConfig(PretrainedConfig):
    model_type = "ldqa"

    def __init__(self, model_type="longformer", **kwargs):
        super().__init__(**kwargs)
        if model_type != "longformer":
            raise ValueError("Only longformer is supported")
        self.model_type = model_type


class LDQAModel(PreTrainedModel):
    config_class = LDQAModelConfig

    def __init__(self, config, base_lm, encoder, projection_head):
        super().__init__(config)
        # TODO: WIP. Functions are not implemented yet.
        # self.base_lm = config.get_base_lm()
        # self.encoder = config.get_encoder()
        # self.projection_head = config.get_projection_head()
        self.base_lm = base_lm
        self.encoder = encoder
        self.projection_head = projection_head
        self.loss_fct = nn.CrossEntropyLoss()
        self.max_chunks_for_doc = 150

    def forward(
        self,
        query_ids,
        query_attention_mask=None,
        document_ids=None,
        document_attention_mask=None,
        document_encoding_outputs=None,
        global_attention_mask=None,
        label_ids=None,
    ):
        """Performs a forward pass through the model. Returns loss and logits if labels are provided else returns logits only.
        :param query_ids: torch.LongTensor of shape [batch_size, query_length]
        :param document_ids: torch.LongTensor of shape [batch_size, document_length]
        :param query_attention_mask: torch.LongTensor of shape [batch_size, query_length]
        :param document_attention_mask: torch.LongTensor of shape [batch_size, document_length]
        :param document_outputs: torch.LongTensor of shape [batch_size, num_chunks, chunk_length, hidden_size] Tokenized output
        :param global_attention_mask: torch.LongTensor of shape [batch_size, document_length]
        :param labels: torch.LongTensor of shape [batch_size, labels_length]
        :return: loss, logits
        """
        # iterate over document chunks and encode or
        # try reshaping document_ids and document_attention_mask
        # TODO: save document embeddings offline and load them here
        if document_encoding_outputs is None:
            # Document encoding are not already calculated
            document_outputs = []
            with torch.no_grad():
                for i in range(document_ids.shape[1]):
                    chunk_document_ids = document_ids[:, i]
                    chunk_document_attention_mask = document_attention_mask[:, i]
                    document_output = self.encoder(
                        chunk_document_ids,
                        attention_mask=chunk_document_attention_mask,
                        global_attention_mask=global_attention_mask,
                    )
                    # document_output.last_hidden_state.shape = [batch_size, chunk_length, hidden_size]
                    document_outputs.append(document_output.last_hidden_state)
                document_outputs = torch.stack(document_outputs, dim=1)
            # document_outputs.shape = [batch_size, num_chunks, chunk_length, hidden_size]
        # pass encoded document to projection head
        else:
            document_outputs = document_encoding_outputs

        # shape of document_outputs before project_head = (batch_size, num_chunks, chunk_length, hidden_size)
        document_outputs = self.projection_head(document_outputs)

        attention_mask = self.prepare_attention_mask(document_outputs)

        # pass encoded document and query to base-lm
        base_lm_outputs = self.base_lm(
            query_ids,
            labels=label_ids,
            attention_mask=query_attention_mask,
            encoder_cross_attn_inputs=document_outputs,
            encoder_cross_attn_attention_masks=attention_mask,
        )

        return base_lm_outputs

    def prepare_attention_mask(self, document_outputs):
        # Shape of document_outputs after pooling projection head  = (batch_size, num_chunks, hidden_size
        num_chunks = document_outputs.shape[1]
        attention_mask = torch.full(  # consider all tokens with projection head
            size=(document_outputs.shape[0], self.max_chunks_for_doc),
            fill_value=-3.4028e38,  # -inf
            dtype=document_outputs.dtype,
            device=document_outputs.device,
        )
        attention_mask[:, :num_chunks] = 0
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        return attention_mask

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )


class EncoderOnlyModelConfig(PretrainedConfig):
    model_type = "encoder_only"

    def __init__(self, model_type="longformer", **kwargs):
        super().__init__(**kwargs)
        if model_type != "longformer":
            raise ValueError("Only longformer is supported")
        self.model_type = model_type


class EncoderOnlyModel(PreTrainedModel):
    """Class for encoder only model. This model is used to precompute document embeddings."""
    config_class = EncoderOnlyModelConfig

    def __init__(self, config, encoder):
        super().__init__(config)
        self.encoder = encoder

    def forward(
        self,
        document_ids,
        document_attention_mask=None,
        global_attention_mask=None,
    ):
        """Performs a forward pass through the model. Returns loss and logits if labels are provided else returns logits only.
        :param document_ids: torch.LongTensor of shape [batch_size, document_length]
        :param document_attention_mask: torch.LongTensor of shape [batch_size, document_length]
        :param global_attention_mask: torch.LongTensor of shape [batch_size, document_length]
        :return: loss, logits
        """
        with torch.inference_mode():
            document_outputs = self.encoder(
                document_ids,
                document_attention_mask,
                global_attention_mask=global_attention_mask,
            )
        return document_outputs.last_hidden_state
