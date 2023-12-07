import argparse

from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import wandb
from dataset import SQuAD_Dataset
from metrics import MetricComputer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--total_steps", type=int, default=32000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args


def setup_dataset(tokenizer, debug=False):
    """Setup dataset object and data collator."""
    dataset_object = SQuAD_Dataset(
        tokenizer=tokenizer,
        split=None,
        streaming=False,
        debug=debug,
    )

    train_dataset = dataset_object.dataset["train"]
    val_dataset = dataset_object.dataset["validation"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="max_length",
        return_tensors="pt",
    )

    return train_dataset, val_dataset, data_collator


if __name__ == "__main__":
    # parse arguments and print to console
    args = parse_args()
    wandb.init(project="huggingface", entity="adv-nlp-ldqa", config=args)
    if args.run_name is not None:
        wandb.run.name = args.run_name
    print(args)

    model_tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

    # set up datasets and data collator
    train_dataset, val_dataset, data_collator = setup_dataset(
        model_tokenizer, args.debug
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "allenai/led-base-16384", use_cache=True
    )

    # set generate hyperparameters
    model.config.num_beams = 4
    model.config.max_length = 40
    model.config.min_length = 2
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    total_steps = args.total_steps
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,  # Adjust based on your requirements
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=total_steps,
        remove_unused_columns=True,
        logging_strategy="steps",
        logging_steps=100,
        dataloader_num_workers=20,
        fp16=True,  # fp16 training
        evaluation_strategy="steps",
        predict_with_generate=True,
        report_to="wandb",
        eval_steps=1000,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=1,
        optim="adamw_apex_fused",
    )

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model_tokenizer,
        data_collator=data_collator,
        compute_metrics=MetricComputer(model_tokenizer),
    )

    trainer.train()
