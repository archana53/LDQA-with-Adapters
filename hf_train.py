import argparse
import itertools

import evaluate
import nltk
import numpy as np
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    LEDForConditionalGeneration,
    Trainer,
    TrainingArguments,
    get_scheduler,
)

from dataset import DataCollatorForLDQA, MuLD_Dataset, TweetQA_Dataset
from encoder import EncoderType
from model import LDQAModel, LDQAModelConfig
from projection_heads import ProjectionHeadType


def parse_args():
    parser = argparse.ArgumentParser()

    train = parser.add_argument_group("Training")
    train.add_argument("--batch_size", type=int, default=1)
    train.add_argument("--total_steps", type=int, default=32000)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight_decay", type=float, default=0.01)
    train.add_argument("--warmup_steps", type=int, default=0)
    train.add_argument("--save_steps", type=int, default=1000)
    train.add_argument("--save_total_limit", type=int, default=2)
    train.add_argument("--output_dir", type=str, default="./output")

    lm = parser.add_argument_group("LM")
    lm.add_argument(
        "--encoder_type",
        type=str,
        default="LongFormer",
        choices=["LongFormer", "LongT5", "LLaMa"],
    )

    projection = parser.add_argument_group("Projection")
    projection.add_argument("--proj_input_dim", type=int, default=768)
    projection.add_argument("--proj_output_dim", type=int, default=768)
    projection.add_argument("--proj_num_self_attention_heads", type=int, default=2)
    projection.add_argument("--proj_num_cross_attention_heads", type=int, default=2)
    projection.add_argument(
        "--proj_type",
        type=str,
        default="AvgPool",
        choices=["AvgPool", "MaxPool", "Attention", "QueryAware"],
    )
    args = parser.parse_args()

    # split args into separate dicts
    arg_groups = {}
    for group in parser._action_groups:
        if group.title in ["positional arguments", "optional arguments", "options"]:
            continue
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    return arg_groups


def print_args(**kwargs):
    """Print dicts of arguments to console."""
    for k, v in kwargs.items():
        print(k.center(48, "-"))
        for arg in vars(v):
            print(f"\t{arg}: {getattr(v, arg)}")
        print("-" * 48)


if __name__ == "__main__":
    # parse arguments and print to console
    all_args = parse_args()
    print_args(**all_args)
    train_args = all_args["Training"]
    lm_args = all_args["LM"]
    projection_args = all_args["Projection"]

    model_tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

    tweetqa_object = TweetQA_Dataset(tokenizer=model_tokenizer, split=None, streaming=True, chunk_size=4096)
    train_dataset = tweetqa_object.dataset["train"]
    val_dataset = tweetqa_object.dataset["validation"]

    data_collator = DataCollatorForLDQA(
        tokenizer=model_tokenizer,
        padding="max_length",
        max_query_length=4096,
        return_tensors="pt",
    )

    # set up base-lm and document encoder
    model_original = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
    base_lm = LEDForConditionalGeneration(model_original.config, cross_attn_encoder=True)
    base_lm.load_state_dict(model_original.state_dict(), strict=False)

    encoder_config = EncoderType[lm_args.encoder_type].value()
    encoder = encoder_config.get_model()

    # set up projection head
    projection_head_config = ProjectionHeadType[projection_args.proj_type].value
    projection_head_config = projection_head_config.from_kwargs(
        input_dim=projection_args.proj_input_dim,
        output_dim=projection_args.proj_output_dim,
        num_self_attention_heads=projection_args.proj_num_self_attention_heads,
        num_cross_attention_heads=projection_args.proj_num_cross_attention_heads,
    )
    projection_head = projection_head_config.get_projection_head()

    # set up LDQA model
    model_config = LDQAModelConfig()
    model = LDQAModel(model_config, base_lm, encoder, projection_head)

    # set up generation config
    generation_config = model_original.generation_config
    generation_config.bos_token_id = model_tokenizer.bos_token_id

    # set up metrics using huggingface evaluate
    nltk.download("punkt", quiet=True)
    metric = evaluate.combine(["rouge", "meteor", "bleu"])

    def get_metrics(eval_prediction):
        """Compute metrics: BLEU, ROUGE, METEOR and return a dictionary.
        :param eval_prediction: instance of EvalPrediction with predictions and labels
        :return: dictionary of metrics
        """
        preds, labels = eval_prediction

        # decode preds and labels
        labels = np.where(labels != -100, labels, model_tokenizer.pad_token_id)
        decoded_preds = model_tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = model_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return result

    total_steps = train_args.total_steps
    training_args = TrainingArguments(
        output_dir=train_args.output_dir,
        num_train_epochs=3,  # Adjust based on your requirements
        per_device_train_batch_size=train_args.batch_size,
        per_device_eval_batch_size=train_args.batch_size,
        save_steps=train_args.save_steps,
        save_total_limit=train_args.save_total_limit,
        max_steps=total_steps,
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=100,
        dataloader_num_workers=4,
    )

    trainable_params = []
    trainable_mod_names = []

    # Only cross attention parameters trainable
    trainable_param_count = 0
    for name, module in model.named_modules():
        if name.endswith("cross") or name.endswith("projection"):
            print(name)
            trainable_params.append(module.parameters())
            trainable_param_count += sum(p.numel() for p in module.parameters())

    # print parameter summaries
    print("-" * 48)
    print("Parameter Summary".center(48, "-"))
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
    print(f"Base LM parameters: {sum(p.numel() for p in base_lm.parameters())}")
    print(f"Projection head parameters: {sum(p.numel() for p in projection_head.parameters())}")
    print(f"Trainable parameters: {trainable_param_count}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print("-" * 48)

    trainable_params = itertools.chain(*trainable_params)
    optimizer = AdamW(trainable_params, lr=train_args.lr, weight_decay=train_args.weight_decay)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=train_args.warmup_steps,
        num_training_steps=total_steps,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model_tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler),
        # compute_metrics=get_metrics,  # TODO: set up evaluation
    )

    trainer.train()
