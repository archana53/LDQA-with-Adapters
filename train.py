from dataset import MuLD_Dataset
import lightning.pytorch as pl
from torch import optim, utils
from torch.utils.data import DataLoader
from transformers import (
    LongformerModel,
    RobertaTokenizer,
    LongformerForQuestionAnswering,
    AutoTokenizer,
)
from transformers import LongformerConfig
import argparse

from encoder import Encoder, EncoderType


class LongDocumentPLM(pl.LightningModule):
    def __init__(self, model, tokenizer, long_encoder):
        super().__init__()

        # Load the existing weights of the longformer model with strict = False
        self.model = model
        self.tokenizer = tokenizer
        self.long_encoder = long_encoder

    def forward(self, document, query):
        encoder_outputs = self.long_encoder.encode(document)
        tokenized_query = self.tokenizer(
            query, return_tensors="pt", return_attention_mask=True, padding=True
        )
        model_outputs = self.model(
            tokenized_query.input_ids,
            attention_mask=tokenized_query.attention_mask,
            cross_modality_inputs=encoder_outputs.last_hidden_state,
            cross_modality_attention_mask=encoder_outputs.attention_mask,
        )
        return model_outputs

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        print(batch.keys())
        document, query, output = batch["document"], batch["query"], batch["output"]
        model_outputs = self.forward(document, query)

        # Logging to TensorBoard (if installed) by default

    def configure_optimizers(self):
        # Only select parameters of cross attention and cross attention output as trainable
        trainable_params = [
            param for name, param in self.model.named_parameters() if "cross" in name
        ]
        optimizer = optim.Adam(trainable_params, lr=1e-3)
        return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="LongFormer",
        choices=["LongFormer", "LongT5", "LLaMa"],
    )
    args = parser.parse_args()

    ## Check this part again!!!

    # Get original pretrained/finetuned weights of the original PLM
    original_longformer = LongformerForQuestionAnswering.from_pretrained(
        "allenai/longformer-base-4096"
    )
    base_weights = original_longformer.state_dict()

    # Replace .attention.output weights with attention.self_output
    new_base_weights = base_weights.copy()
    for key in base_weights.keys():
        if "attention.output" in key:
            value = base_weights[key]
            new_base_weights.remove(key)
            new_base_weights[key.replace("attention.output", "attention.self_output")]

    # Load the modified model version and copy all other weights except cross attention module

    config = LongformerConfig()
    # Modify the config to match that of the pretrained model that is available

    # set up base-lm and document encoder
    model = LongformerModel(config, cross_modality=True)
    model.load_state_dict(new_base_weights, strict=False)
    model_tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    encoder = Encoder(EncoderType.LongFormer)

    ldplm = LongDocumentPLM(model, model_tokenizer, encoder)

    args = parser.parse_args()
    muld_object = MuLD_Dataset()
    # train_dataset = muld_object.get_dataset(split = 'train', streaming = True).with_format('torch')
    val_dataset = muld_object.get_dataset(
        split="validation", streaming=True
    ).with_format("torch")
    # test_dataset = muld_object.get_dataset(split = 'test', streaming= True).with_format('torch')

    train_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=ldplm, train_dataloaders=train_dataloader)
