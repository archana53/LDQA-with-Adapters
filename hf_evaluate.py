import argparse
import os

import evaluate
import nltk
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LEDForConditionalGeneration

from dataset import MuLD_Dataset, TweetQA_Dataset
from encoder import EncoderType
from model import LDQAModel, LDQAModelConfig
from projection_heads import ProjectionHeadType


def parse_args():
    parser = argparse.ArgumentParser()

    checkpoint = parser.add_argument_group("Checkpoint")
    checkpoint.add_argument("--checkpoint_dir", type=str, default="./output")

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
        choices=["AvgPool", "MaxPool", "Linear", "Attention", "QueryAware"],
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
    checkpoint_args = all_args["Checkpoint"]
    lm_args = all_args["LM"]
    projection_args = all_args["Projection"]

    # set up tokenizer
    model_tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

    # set up validation dataset
    dataset_obj = TweetQA_Dataset(
        tokenizer=model_tokenizer, split="validation", streaming=False, chunk_size=4096
    )
    val_dataset = dataset_obj.dataset.map(dataset_obj.tokenize)

    # set up base LM
    model_original = LEDForConditionalGeneration.from_pretrained(
        "allenai/led-base-16384"
    )
    base_lm = LEDForConditionalGeneration(
        model_original.config, cross_attn_encoder=True
    )

    # set up encoder
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
    )
    print(model_config)

    # load LDQA model from checkpoint
    model.load_state_dict(
        torch.load(os.path.join(checkpoint_args.checkpoint_dir, "pytorch_model.bin"))
    )
    model = model.to("cuda")

    # set generate hyperparameters
    model.base_lm.config.num_beams = 4
    model.base_lm.config.max_length = 40
    model.base_lm.config.min_length = 2
    model.base_lm.config.length_penalty = 2.0
    model.base_lm.config.early_stopping = True
    model.base_lm.config.no_repeat_ngram_size = 3

    # set up metrics using huggingface evaluate
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    def generate_answer(batch):
        query_ids = batch["query_ids"].to("cuda")
        query_attention_mask = batch["query_attention_mask"].to("cuda")
        document_ids = batch["document_ids"].to("cuda")
        document_attention_mask = batch["document_attention_mask"].to("cuda")

        document_ids = document_ids.unsqueeze(1)
        document_attention_mask = document_attention_mask.unsqueeze(1)

        # create global attention mask
        global_attention_mask = torch.ones_like(
            document_attention_mask, device=document_attention_mask.device
        )

        preds = model.generate(
            query_ids=query_ids,
            query_attention_mask=query_attention_mask,
            document_ids=document_ids,
            document_attention_mask=document_attention_mask,
            global_attention_mask=global_attention_mask,
        )

        # decode preds and labels
        labels = np.where(
            batch["label_ids"] != -100, batch["label_ids"], model_tokenizer.pad_token_id
        )
        decoded_preds = model_tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = model_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = [
            "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
        ]
        decoded_labels = [
            "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
        ]

        rouge_score = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        meteor_score = meteor.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        # for bleu convert the references to a list of lists
        decoded_labels = [[label] for label in decoded_labels]
        bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)

        # expand n-gram precisions list to a dictionary
        bleu_score.update(
            {
                f"precision_{i}": score
                for i, score in enumerate(bleu_score["precisions"])
            }
        )
        del bleu_score["precisions"]
        # prefix all keys with bleu
        bleu_score = {f"bleu_{k}": v for k, v in bleu_score.items()}

        # combine all metrics into one dictionary
        results = {
            k: v
            for score_dict in [rouge_score, bleu_score, meteor_score]
            for k, v in score_dict.items()
        }
        return results

    # accumulate metrics over the validation dataset
    results = {}
    for batch in tqdm(val_dataset):
        batch_results = generate_answer(batch)
        for k, v in batch_results.items():
            if k not in results:
                results[k] = []
            results[k].append(v)

    # average metrics over the validation dataset
    for k, v in results.items():
        results[k] = sum(v) / len(v)

    # print metrics to the console
    print("Validation results:")
    for k, v in results.items():
        print(f"{k}: {v}")
