import argparse
import os
import h5py

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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    group.add_argument("--force", action="store_true", help="force overwrite existing embeddings")
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

    muld_object = MuLD_Dataset(tokenizer=model_tokenizer, split=None, streaming=True, chunk_size=4096)
    train_dataset = muld_object.dataset["train"].map(muld_object.tokenize)
    val_dataset = muld_object.dataset["validation"].map(muld_object.tokenize)
    test_dataset = muld_object.dataset["test"].map(muld_object.tokenize)

    # set up h5py file handles for storing precomputed embeddings
    if not os.path.exists(args.dest):
        os.mkdir(args.dest)
    h5py_dest = os.path.join(args.dest, "embeddings.h5")
    already_exists = True if os.path.exists(h5py_dest) else False

    with h5py.File(h5py_dest, "a", swmr=True) as embedding_store:
        # if resume, load keys from the embedding store
        if args.force:
            print("Overwriting existing embeddings")
            embedding_store.clear()
        elif already_exists and not args.resume:
            raise ValueError(
                "Embedding store already exists. Use --resume to resume"
                "from the existing checkpoint or --force to overwrite."
            )

        # iterate over dataset and store document embeddings
        for dataset in [train_dataset, val_dataset, test_dataset]:
            for example in tqdm(dataset):
                document = example["document"]

                # skip if document already in embedding store
                if document in embedding_store:
                    continue

                document_ids = example["document_ids"]
                document_attention_mask = example["document_attention_mask"]
                global_attention_mask = None
                document_embeddings = model(
                    document_ids,
                    document_attention_mask=document_attention_mask,
                    global_attention_mask=global_attention_mask,
                )

                # store document embeddings
                embedding_store.create_dataset(document, data=document_embeddings.numpy())
