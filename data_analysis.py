from argparse import ArgumentParser

import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
from transformers import AutoTokenizer

from configs import DATASET_CONFIG


def get_lengths(batch):
    query_len = np.array(batch["query_attention_mask"]).sum(axis=1)
    document_len = np.array(batch["document_attention_mask"]).sum(axis=1)
    label_len = np.array(batch["label_attention_mask"]).sum(axis=1)

    lengths = {"query": query_len, "document": document_len, "label": label_len}
    return lengths


def collect_stats(dataset):
    stats = {"query": [], "document": [], "label": []}
    for batch in tqdm(dataset):
        for key in stats:
            stats[key].extend(batch[key])
    return stats


def print_statistics(lengths):
    def _get_row(key):
        return [
            f"{key.capitalize()} length",
            lengths[key].min().item(),
            lengths[key].max().item(),
            lengths[key].float().mean().item(),
            np.median(lengths[key]),
            np.percentile(lengths[key], 90),
            np.percentile(lengths[key], 95),
            np.percentile(lengths[key], 99),
        ]

    table = PrettyTable()
    table.field_names = [
        "",
        "Min",
        "Max",
        "Mean",
        "Median",
        "90th percentile",
        "95th percentile",
        "99th percentile",
    ]
    table.add_row(_get_row("query"))
    table.add_row(_get_row("document"))
    table.add_row(_get_row("label"))
    print(table)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="TweetQA", choices=["TweetQA", "SQuAD"]
    )
    args = parser.parse_args()

    # set up tokenizer
    model_tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

    # set up dataset
    STREAMING = False
    dataset_obj = DATASET_CONFIG[args.dataset]["cls"](
        tokenizer=model_tokenizer,
        split=None,
        streaming=STREAMING,
        chunk_size=4096,
        mode="ldqa",
    )
    train_dataset = dataset_obj.dataset["train"].map(
        dataset_obj.tokenize, batched=True, batch_size=1024
    )
    val_dataset = dataset_obj.dataset["validation"].map(
        dataset_obj.tokenize, batched=True, batch_size=1024
    )

    train_lengths = train_dataset.map(get_lengths, batched=True, batch_size=1024)
    val_lengths = val_dataset.map(get_lengths, batched=True, batch_size=1024)

    if STREAMING:
        # iterate over streaming datasets to get statistics
        train_lengths = collect_stats(train_lengths)
        val_lengths = collect_stats(val_lengths)

    print("Train statistics:")
    print_statistics(train_lengths)
    print()

    print("Validation statistics:")
    print_statistics(val_lengths)
