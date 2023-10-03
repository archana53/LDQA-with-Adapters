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

    def __init__(self, config):
        super().__init__(config)
        if config.model_type == "longformer":
            # TODO: WIP. Functions are not implemented yet.
            self.base_lm = get_longformer_with_cross_attention()
            self.encoder = get_longformer_encoder()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        query_ids,
        document_ids,
        query_attention_mask=None,
        document_attention_mask=None,
        global_attention_mask=None,
        labels=None,
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
        # get encoded document
        document_outputs = self.encoder(
            document_ids,
            attention_mask=document_attention_mask,
            global_attention_mask=global_attention_mask,
        )

        # pass encoded document and query to base-lm
        base_lm_outputs = self.base_lm(
            query_ids,
            attention_mask=query_attention_mask,
            cross_modality_inputs=document_outputs.last_hidden_state,
            cross_modality_attention_mask=document_outputs.attention_mask,
        )

        # compute loss if labels are provided
        if labels is not None:
            loss = self.loss_fct(base_lm_outputs.logits, labels)
            return loss, base_lm_outputs.logits
        else:
            return base_lm_outputs.logits
