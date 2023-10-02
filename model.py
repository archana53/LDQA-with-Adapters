
import torch
from torch import nn
import torch.nn.functional as F
from transformers import LongformerModel, RobertaTokenizer
from transformers import LongformerConfig
from torchsummary import summary


def add_cross_attention_modules(model):
    print(model)


def pad_to_window_size(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       one_sided_window_size: int, pad_token_id: int):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer selfattention.
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = int(2 * one_sided_window_size)
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
    return input_ids, attention_mask

class BaseModel(object):
    def __init__(self,**kwargs):
        super(BaseModel, self).__init__(**kwargs)

class LongFormer(BaseModel):
    def __init__(self,**kwargs):
        super(LongFormer, self).__init__(**kwargs)
        self.hf_model_uuid = "allenai/longformer-base-4096"
        self.model = LongformerModel.from_pretrained(self.hf_model_uuid)

    def get_model(self):
        return self.model
    def forward(self,x):
        self.model(x)


if __name__ == '__main__':
    config = LongformerConfig()
    model = LongformerModel(config, cross_modality = True)
    add_cross_attention_modules(model)

    """
    model = model.model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.model_max_length = model.config.max_position_embeddings

    SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document

    input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

    # TVM code doesn't work on CPU. Uncomment this if `config.attention_mode = 'tvm'`
    # model = model.cuda(); input_ids = input_ids.cuda()

    # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
    attention_mask[:, [1, 4, 21,]] =  2  # Set global attention based on the task. For example,
                                        # classification: the <s> token
                                        # QA: question tokens

    # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
    input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, config.attention_window[0], tokenizer.pad_token_id)

    output = model(input_ids, attention_mask=attention_mask)[0]
    """