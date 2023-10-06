import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


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

    def forward(
        self,
        query_ids,
        document_ids,
        query_attention_mask=None,
        document_attention_mask=None,
        global_attention_mask=None,
        label_ids=None,
    ):
        """Performs a forward pass through the model. Returns loss and logits if labels are provided else returns logits only.
        :param query_ids: torch.LongTensor of shape [batch_size, query_length]
        :param document_ids: torch.LongTensor of shape [batch_size, document_length]
        :param query_attention_mask: torch.LongTensor of shape [batch_size, query_length]
        :param document_attention_mask: torch.LongTensor of shape [batch_size, document_length]
        :param global_attention_mask: torch.LongTensor of shape [batch_size, document_length]
        :param labels: torch.LongTensor of shape [batch_size, labels_length]
        :return: loss, logits
        """
        # iterate over document chunks and encode or
        # try reshaping document_ids and document_attention_mask
        # TODO: save document embeddings offline and load them here
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
                document_outputs.append(document_output)
            document_outputs = torch.cat(document_outputs, dim=1)

        # pass encoded document to projection head
        document_outputs = self.projection_head(document_outputs.last_hidden_state)
        attention_mask = torch.ones(  # consider all tokens with projection head
            document_outputs.shape[0],
            document_outputs.shape[1],
            dtype=document_attention_mask.dtype,
        )

        # pass encoded document and query to base-lm
        base_lm_outputs = self.base_lm(
            query_ids,
            attention_mask=query_attention_mask,
            cross_modality_inputs=document_outputs.last_hidden_state,
            cross_modality_attention_mask=attention_mask,
        )

        # compute loss if labels are provided
        if label_ids is not None:
            loss = self.loss_fct(base_lm_outputs.logits, label_ids)
            return loss, base_lm_outputs.logits
        else:
            return base_lm_outputs.logits
