import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


class LDQAModelConfig(PretrainedConfig):
    model_type = "LQDA"

    def __init__(self, model_type="LDQA", **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type


class LDQAModel(PreTrainedModel):
    config_class = LDQAModelConfig

    def __init__(
        self, config, base_lm, encoder, projection_head, max_chunks_for_doc=1
    ):
        super().__init__(config)
        self.base_lm = base_lm
        self.encoder = encoder
        self.projection_head = projection_head
        self.loss_fct = nn.CrossEntropyLoss()
        self.max_chunks_for_doc = max_chunks_for_doc

    def forward(
        self,
        query_ids,
        query_attention_mask=None,
        document_ids=None,
        document_attention_mask=None,
        document_encoding_outputs=None,
        global_attention_mask=None,
        labels=None,
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
            document_outputs = self.encode_document(
                document_ids, document_attention_mask, global_attention_mask
            )
            # document_outputs.shape = [batch_size, num_chunks, chunk_length, hidden_size]
        else:
            document_outputs = document_encoding_outputs

        # pass encoded document to projection head
        # shape of document_outputs before project_head = (batch_size, num_chunks, chunk_length, hidden_size)
        document_outputs = self.projection_head(
            document_outputs, x_mask=document_attention_mask
        )

        attention_mask = self._prepare_attention_mask(document_attention_mask)

        # pass encoded document and query to base-lm
        base_lm_outputs = self.base_lm(
            query_ids,
            labels=labels,
            attention_mask=query_attention_mask,
            encoder_cross_attn_inputs=document_outputs,
            encoder_cross_attn_attention_masks=attention_mask,
        )

        return base_lm_outputs

    @torch.no_grad()
    def encode_document(
        self, document_ids, document_attention_mask, global_attention_mask
    ):
        """Encodes a document using the encoder and returns the document embeddings.
        :param document_ids: torch.LongTensor of shape [batch_size, document_length]
        :param document_attention_mask: torch.LongTensor of shape [batch_size, document_length]
        :param global_attention_mask: torch.LongTensor of shape [batch_size, document_length]
        :return: torch.FloatTensor of shape [batch_size, num_chunks, chunk_length, hidden_size]
        """

        document_outputs = []
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
        return document_outputs

    def _prepare_attention_mask(self, document_attention_mask, repeat=1):
        """Creates cross attention mask for base-lm. Takes the document_attention_mask
        and returns a mask for post-projection operations. Chunks with at least one
        non-padded token are considered valid.

        :param document_attention_mask: torch.LongTensor of shape [batch_size, num_chunks, chunk_length]
        :param repeat: int, number of times to repeat the mask per chunk
        useful when projection head returns multiple outputs per chunk
        :return: torch.LongTensor of shape [batch_size, 1, 1, num_chunks]
        """
        cross_attn_attention_mask = document_attention_mask.any(dim=-1).long()
        cross_attn_attention_mask = (
            cross_attn_attention_mask.repeat_interleave(repeat, dim=-1)
        )
        cross_attn_attention_mask = cross_attn_attention_mask.unsqueeze(
            1
        ).unsqueeze(2)
        return cross_attn_attention_mask

    def generate(
        self,
        query_ids=None,
        query_attention_mask=None,
        document_ids=None,
        document_attention_mask=None,
        document_encoding_outputs=None,
        attention_mask=None,
        global_attention_mask=None,
        **model_kwargs,
    ):
        if document_encoding_outputs is None:
            # Document encoding are not already calculated
            document_outputs = self.encode_document(
                document_ids, document_attention_mask, global_attention_mask
            )
            # document_outputs.shape = [batch_size, num_chunks, chunk_length, hidden_size]
        else:
            document_outputs = document_encoding_outputs

        # shape of document_outputs before project_head = (batch_size, num_chunks, chunk_length, hidden_size)
        document_outputs = self.projection_head(
            document_outputs, x_mask=document_attention_mask
        )

        attention_mask = self._prepare_attention_mask(document_attention_mask)

        return self.base_lm.generate(
            inputs=query_ids,
            attention_mask=query_attention_mask,
            encoder_cross_attn_inputs=document_outputs,
            encoder_cross_attn_attention_masks=attention_mask,
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
        """Performs a forward pass through the model.
        Returns loss and logits if labels are provided else returns logits only.
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
