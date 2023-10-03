from dataset import MuLD_Dataset
from torch import optim, utils
from torch.utils.data import DataLoader
from transformers import (
    LongformerModel,
    RobertaTokenizer,
    LongformerForQuestionAnswering,
    AutoTokenizer,
    DataCollatorForSeq2Seq
)
from transformers import Trainer, TrainingArguments
from transformers import LongformerConfig
from transformers import get_scheduler
import argparse
from torch.optim import Adam
from encoder import Encoder, EncoderType
import itertools

#breakpoint()


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
    model = LongformerModel(model_original.config, cross_modality=True)
    model.load_state_dict(model_original.state_dict(), strict = False)

    model_tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    encoder = Encoder(EncoderType.LongFormer)
    
    muld_object = MuLD_Dataset()
    val_dataset = muld_object.get_dataset(
        split="validation", streaming=True
    )

    # Tokenize your dataset using the tokenizer
    def tokenize_function(examples):
        return model_tokenizer(examples["input"], padding="max_length", truncation=True)

    tokenized_datasets = val_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=model_tokenizer, padding = 'max_length', max_length = 4096 )

    training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,  # Adjust based on your requirements
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
    )

    trainable_params = []
    trainable_mod_names = []

    #Only cross attention parameters trainable
    for name,module in model.named_modules():
        if 'cross' in name:
            trainable_params.append(module.parameters())
    
    trainable_params = (itertools.chain(*trainable_params))
    print(trainable_params)
    optimizer = Adam(trainable_params, lr=1e-3)

    total_steps = len(list(tokenized_datasets)) * training_args.num_train_epochs / 8
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    trainer = Trainer(
    model,
    training_args,
    train_dataset=val_dataset,
    eval_dataset=val_dataset,
    tokenizer=model_tokenizer,
    )

    trainer.train()
    