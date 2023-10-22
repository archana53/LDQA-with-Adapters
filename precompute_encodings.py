import argparse
import os
import json

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import MuLD_Dataset
from encoder import EncoderType
from model import EncoderOnlyModelConfig, EncoderOnlyModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="LongFormer",
        choices=["LongFormer", "LongT5", "LLaMa"],
    )
    parser.add_argument("--dest", type=str, help="destination directory for new dataset")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # parse arguments and print to console
    args = parse_args()

    # set up document encoder
    model_tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    encoder_config = EncoderType[args.encoder_type].value()
    encoder = encoder_config.get_model()

    # set up EncoderOnly model
    model_config = EncoderOnlyModelConfig()
    model = EncoderOnlyModel(model_config, encoder) 

    muld_object = MuLD_Dataset(
        tokenizer=model_tokenizer, split=None, streaming=True, chunk_size=4096
    )
    train_dataset = muld_object.dataset["train"].map(muld_object.tokenize)
    val_dataset = muld_object.dataset["validation"].map(muld_object.tokenize)
    test_dataset = muld_object.dataset["test"].map(muld_object.tokenize)

    # set up json file handles for storing precomputed embeddings
    if not os.path.exists(args.dest):
        os.mkdir(args.dest)
        
    train_json = open(os.path.join(args.dest, "train.json"), "w", encoding="utf-8")
    val_json = open(os.path.join(args.dest, "val.json"), "w", encoding="utf-8")
    test_json = open(os.path.join(args.dest, "test.json"), "w", encoding="utf-8")

    # store unique documents to avoid recomputing embeddings
    unique_documents = {}

    # iterate over dataset and store document embeddings
    for dataset, json_handler in [(train_dataset, train_json), (val_dataset, val_json), (test_dataset, test_json)]:
        for example in tqdm(dataset):
            document = example["document"]
            
            if document in unique_documents:
                example[f"{args.encoder_type}_document_embeddings"] = unique_documents[document]
                json_handler.write(json.dumps(example) + "\n")
                continue

            document_ids = example["document_ids"]
            document_attention_mask = example["document_attention_mask"]
            global_attention_mask = None
            document_embeddings = model(
                document_ids,
                document_attention_mask=document_attention_mask,
                global_attention_mask=global_attention_mask,
            )
            example[f"{args.encoder_type.lower()}_document_embeddings"] = document_embeddings
            
            # convert tensors to lists for json serialization
            for key, value in example.items():
                if isinstance(value, torch.Tensor):
                    example[key] = value.tolist()
            json_handler.write(json.dumps(example) + "\n")

    # close json file handles
    train_json.close()
    val_json.close()
    test_json.close()
