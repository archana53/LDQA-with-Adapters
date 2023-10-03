import re

from datasets import load_dataset, load_from_disk


class HFDataset:
    def __init__(self, dataset_uuid=None, **kwargs):
        self.dataset_uuid = dataset_uuid

    def get_dataset(self) -> dict:
        """Returns the corresponding dataset from huggingface"""
        raise NotImplementedError


class MuLD_Dataset(HFDataset):
    def __init__(
        self, tokenizer, split="train", chunk_size=4096, streaming=False, **kwargs
    ):
        super(MuLD_Dataset, self).__init__(dataset_uuid="ghomasHudson/muld", **kwargs)
        CACHE_PATH = "~/muld_dataset"

        try:
            self.dataset = load_from_disk(CACHE_PATH)
        except FileNotFoundError:
            print("Dataset not found, loading from huggingface")
            dataset = self.get_dataset(split=split, streaming=streaming)
            dataset = dataset.map(self.preprocess)
            dataset = dataset.remove_columns(["input", "metadata"])
            dataset = dataset.rename_column("output", "label")
            dataset = dataset.map(self.tokenize, batched=False)
            # dataset.save_to_disk(CACHE_PATH)  # TODO: unable to save IterableDataset
            self.dataset = dataset

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def get_dataset(self, split="train", streaming=False) -> dict:
        dataset = load_dataset(
            self.dataset_uuid, "NarrativeQA", split=split, streaming=streaming
        ).with_format("torch")
        return dataset

    def preprocess(self, example: dict) -> dict:
        """Preprocess the dataset with basic cleanup and splitting into query and document"""

        # remove BOM tags: '\u00ef\u00bb\u00bf' and HTML tags using regex
        input = example["input"].replace("\u00ef\u00bb\u00bf", "")
        input = re.sub(r"<.*?>", "", input)
        # remove newlines and extra spaces
        input = input.replace("\n", " ").strip()

        # splits input into query and document
        query, document = input.split("?", 1)
        query = query.strip() + "?"

        return {"query": query, "document": document}

    def tokenize(self, example: dict) -> dict:
        """Tokenize the dataset with the encoder tokenizer"""
        tokenized_query = self.tokenizer(
            example["query"],
            padding=True,
            truncation=False,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )
        tokenized_document = self.tokenizer(
            example["document"],  # TODO: check if each chunk has a BOS token
            padding=True,
            truncation=True,
            max_length=self.chunk_size,
            return_tensors="pt",
            pad_to_multiple_of=8,
            return_overflowing_tokens=True,
        )

        return_dict = {
            "query_ids": tokenized_query.input_ids,
            "query_attention_mask": tokenized_query.attention_mask,
            "document_ids": tokenized_document.input_ids,
            "document_attention_mask": tokenized_document.attention_mask,
        }

        if "label" in example:
            tokenized_output = self.tokenizer(
                example["label"],
                padding=True,
                truncation=False,
                return_tensors="pt",
                pad_to_multiple_of=8,
            )
            return_dict["label_ids"] = tokenized_output.input_ids
            return_dict["label_attention_mask"] = tokenized_output.attention_mask

        return return_dict


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    muld = MuLD_Dataset(streaming=True, split="validation", tokenizer=tokenizer)
    for ex in muld.dataset:
        break
    print(ex.keys())
    print(ex["query_ids"].shape)
    print(ex["document_ids"].shape)
    print(ex["query_attention_mask"].shape)
    print(ex["document_attention_mask"].shape)
