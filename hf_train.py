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

def my_generator(n):
    for i in range(n):
        yield {"input": "sample text", "output": "filler text", "metadata": []}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
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

    # set up base-lm and document encoder
    model_original = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    base_lm = LongformerModel(model_original.config, cross_modality=True)
    base_lm.load_state_dict(model_original.state_dict(), strict=False)

    model_tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
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

    total_steps = train_args.total_steps
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,  # Adjust based on your requirements
        per_device_train_batch_size=train_args.batch_size,
        save_steps=train_args.save_steps,
        save_total_limit=train_args.save_total_limit,
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
    optimizer = Adam(
        trainable_params, lr=train_args.lr, weight_decay=train_args.weight_decay
    )

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
    )

    trainer.train()

