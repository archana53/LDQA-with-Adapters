import argparse
import itertools

from datasets import IterableDataset
from torch import optim, utils
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import MuLD_Dataset
from encoder import EncoderType
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LongformerConfig,
    LongformerForQuestionAnswering,
    LongformerModel,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    get_scheduler,
)

# breakpoint()


def my_generator(n):
    for i in range(n):
        yield {"input": "sample text", "output": "filler text", "metadata": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="LongFormer",
        choices=["LongFormer", "LongT5", "LLaMa"],
    )
    args = parser.parse_args()

    # set up base-lm and document encoder
    model_original = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    model = LongformerModel(model_original.config, cross_attn=True)
    model.load_state_dict(model_original.state_dict(), strict=False)

    print(model.config)

    model_tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    encoder = EncoderType.LongFormer.value()

    muld_object = MuLD_Dataset()
    val_dataset = muld_object.get_dataset(split="validation", streaming=True)

    sample_dataset = IterableDataset.from_generator(my_generator, gen_kwargs={"n": 1})

    # Tokenize your dataset using the tokenizer
    def tokenize_function(examples):
        return model_tokenizer(examples["input"], padding="max_length", truncation=True)

    tokenized_datasets = val_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=model_tokenizer, padding="max_length", max_length=4098
    )

    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,  # Adjust based on your requirements
        per_device_train_batch_size=8,
        save_steps=1000,
        save_total_limit=2,
    )

    trainable_params = []
    trainable_mod_names = []

    # Only cross attention parameters trainable
    for name, module in model.named_modules():
        if "cross" in name:
            trainable_mod_names.append(name)
            trainable_params.append(module.parameters())

    print(trainable_mod_names)
    trainable_params = itertools.chain(*trainable_params)
    optimizer = Adam(trainable_params, lr=1e-3)

    total_steps = (
        len(list(tokenized_datasets)) * training_args.num_train_epochs / args.batch_size
    )
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=sample_dataset,
        eval_dataset=sample_dataset,
        tokenizer=model_tokenizer,
    )

    trainer.train()
