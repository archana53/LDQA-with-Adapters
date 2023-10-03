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
            dataset = dataset.map(self.preprocess).map(self.tokenize, batched=True)
            dataset.save_to_disk(CACHE_PATH)
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
        example["input"] = example["input"].replace("\u00ef\u00bb\u00bf", "")
        example["input"] = re.sub(r"<.*?>", "", example["input"])
        # remove newlines and extra spaces
        example["input"] = example["input"].replace("\n", " ").strip()

        # splits input into query and document
        query, document = example["input"].split("?", 1)
        query = query.strip() + "?"

        # update example
        example["query"] = query
        example["document"] = document
        example["label"] = example["output"]  # rename output to label for hf compatibility

        example.pop("input")
        example.pop("metadata")
        example.pop("output")
        return example

    def tokenize(self, example: dict) -> dict:
        """Tokenize the dataset with the encoder tokenizer"""
        return self.tokenizer(
            example,  #TODO: check if this is correct
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.chunk_size,
            return_tensors="pt",
            pad_to_multiple_of=8,
            return_overflowing_tokens=True,
        )

    def chunk(self, example: dict) -> dict:
        """Split the document into chunks of size chunk_size"""
        document = example["document"]
        chunks = [
            document[i : i + self.chunk_size]
            for i in range(0, len(document), self.chunk_size)
        ]
        example["document"] = chunks
        return example


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("longformer-base-4096")
    muld = MuLD_Dataset(streaming=True, tokenizer=tokenizer)
    print(muld.dataset["train"][0].keys())
    print(muld.dataset["train"][0]["query"])
    print(muld.dataset["train"][0]["output"])
