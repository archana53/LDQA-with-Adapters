import argparse
import itertools

from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    LEDForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    get_scheduler,
)

import wandb
from configs import DATASET_CONFIG
from dataset import DataCollatorForLDQA
from encoder import EncoderType
from metrics import MetricComputer
from model import LDQAModel, LDQAModelConfig
from projection_heads import ProjectionHeadType


def parse_args():
    parser = argparse.ArgumentParser()

    train = parser.add_argument_group("Training")
    train.add_argument(
        "--dataset", type=str, default="MuLD", choices=["MuLD", "TweetQA"]
    )
    train.add_argument("--use_hdf5", action="store_true")
    train.add_argument("--hdf5_path", type=str, default=None)
    train.add_argument("--batch_size", type=int, default=2)
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


def setup_dataset(dataset_config, train_args, tokenizer):
    """Setup dataset object and data collator."""
    dataset_object = dataset_config["cls"](
        tokenizer=tokenizer, split=None, streaming=True, chunk_size=4096
    )

    train_dataset = dataset_object.dataset["train"]
    val_dataset = dataset_object.dataset["validation"]

    hdf5_path = None
    if train_args.use_hdf5:
        # use hdf5 path from args if provided, else use the one from config
        hdf5_path = train_args.hdf5_path or dataset_config["hdf5_path"]

    data_collator = DataCollatorForLDQA(
        tokenizer=tokenizer,
        padding="max_length",
        max_query_length=4096,
        return_tensors="pt",
        hdf5_file_path=hdf5_path,
        max_chunks_for_doc=dataset_config["max_chunks_for_doc"],
    )

    return train_dataset, val_dataset, data_collator


if __name__ == "__main__":
    # parse arguments and print to console
    all_args = parse_args()
    wandb.init(project="huggingface", entity="adv-nlp-ldqa", config=all_args)
    print_args(**all_args)
    train_args = all_args["Training"]
    lm_args = all_args["LM"]
    projection_args = all_args["Projection"]

    model_tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

    # set up datasets and data collator
    dataset_config = DATASET_CONFIG[train_args.dataset]
    train_dataset, val_dataset, data_collator = setup_dataset(
        dataset_config, train_args, model_tokenizer
    )

    # set up base-lm and document encoder
    model_original = LEDForConditionalGeneration.from_pretrained(
        "allenai/led-base-16384"
    )
    base_lm = LEDForConditionalGeneration(
        model_original.config, cross_attn_encoder=True
    )
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
    model = LDQAModel(
        model_config,
        base_lm,
        encoder,
        projection_head,
        max_chunks_for_doc=dataset_config["max_chunks_for_doc"],
    )

    # set up generation config
    generation_config = model_original.generation_config
    generation_config.bos_token_id = model_tokenizer.bos_token_id

    total_steps = train_args.total_steps
    training_args = Seq2SeqTrainingArguments(
        output_dir=train_args.output_dir,
        num_train_epochs=3,  # Adjust based on your requirements
        per_device_train_batch_size=train_args.batch_size,
        per_device_eval_batch_size=1,
        save_steps=train_args.save_steps,
        save_total_limit=train_args.save_total_limit,
        max_steps=total_steps,
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=100,
        dataloader_num_workers=4,
        evaluation_strategy="steps",
        predict_with_generate=True,
        report_to="wandb",
        eval_steps=1000,
    )

    trainable_params = []
    trainable_mod_names = []

    # Only cross attention and projection parameters trainable
    print("-" * 48)
    print("Trainable Parameters".center(48, "-"))
    trainable_param_count = 0
    for name, module in model.named_modules():
        if name.endswith("cross") or name.endswith("projection"):
            print(name)
            trainable_params.append(module.parameters())
            trainable_param_count += sum(p.numel() for p in module.parameters())
    print("-" * 48)

    # print parameter summaries
    print("-" * 48)
    print("Parameter Summary".center(48, "-"))
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters())}")
    print(f"Base LM parameters: {sum(p.numel() for p in base_lm.parameters())}")
    print(
        f"Projection head parameters: {sum(p.numel() for p in projection_head.parameters())}"
    )
    print(f"Trainable parameters: {trainable_param_count}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print("-" * 48)

    trainable_params = itertools.chain(*trainable_params)
    optimizer = AdamW(
        trainable_params, lr=train_args.lr, weight_decay=train_args.weight_decay
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=train_args.warmup_steps,
        num_training_steps=total_steps,
    )

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model_tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler),
        compute_metrics=MetricComputer(model_tokenizer),
    )

    trainer.train()
