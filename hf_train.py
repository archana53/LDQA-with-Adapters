import argparse
import itertools

from torch.optim import Adam
from transformers import (
    AutoTokenizer,
    LongformerModel,
    Trainer,
    TrainingArguments,
    get_scheduler,
)

from dataset import DataCollatorForLDQA, MuLD_Dataset
from encoder import Encoder, EncoderType
from model import LDQAModel, LDQAModelConfig


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

    # set up base-lm and document encoder
    model_original = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    base_lm = LongformerModel(model_original.config, cross_modality=True)
    base_lm.load_state_dict(model_original.state_dict(), strict=False)

    model_tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    encoder = Encoder(EncoderType.LongFormer)

    # set up LDQA model
    model_config = LDQAModelConfig()
    model = LDQAModel(model_config, base_lm, encoder)

    muld_object = MuLD_Dataset(
        tokenizer=model_tokenizer, split=None, streaming=True, chunk_size=4096
    )
    train_dataset = muld_object.dataset["train"]
    val_dataset = muld_object.dataset["validation"]

    data_collator = DataCollatorForLDQA(
        tokenizer=model_tokenizer,
        padding="max_length",
        max_query_length=4096,
        return_tensors="pt",
    )

    total_steps = 32000
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,  # Adjust based on your requirements
        per_device_train_batch_size=2,
        save_steps=1000,
        save_total_limit=2,
        max_steps=total_steps,
        remove_unused_columns=False,
    )

    trainable_params = []
    trainable_mod_names = []

    # Only cross attention parameters trainable
    for name, module in model.named_modules():
        if "cross" in name:
            trainable_params.append(module.parameters())

    trainable_params = itertools.chain(*trainable_params)
    print(trainable_params)
    optimizer = Adam(trainable_params, lr=1e-3)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
