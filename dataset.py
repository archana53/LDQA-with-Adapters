import re
from hashlib import sha256
from typing import List, Optional, Tuple, Union, Dict

import h5py
import preprocessor as tweet_preprocessor
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class HFDataset:
    def __init__(self, dataset_uuid=None, **kwargs):
        self.dataset_uuid = dataset_uuid

    def get_dataset(self) -> dict:
        """Returns the corresponding dataset from huggingface"""
        raise NotImplementedError


class TweetQA_Dataset(HFDataset):
    """HuggingFace TweetQA dataset
    Downloads the dataset from HuggingFace or loads from disk
    Keys in the example:
        `query`: str
        `document`: str
        `label` [optional]: str
        `query_ids`: torch.Tensor of shape (1, query_length)
        `query_attention_mask`: torch.Tensor of shape (1, query_length)
        `document_ids`: torch.Tensor of shape (1, num_chunks, chunk_size)
        `document_attention_mask`: torch.Tensor of shape (1, num_chunks, chunk_size)
        `label_ids`[optional]: torch.Tensor of shape (1, output_length)
        `label_attention_mask` [optional]: torch.Tensor of shape (1, output_length)
    """

    def __init__(
        self, tokenizer, split="train", chunk_size=4096, streaming=False, **kwargs
    ):
        super(TweetQA_Dataset, self).__init__(dataset_uuid="tweet_qa", **kwargs)
        CACHE_PATH = "~/tweetqa_dataset"

        try:
            self.dataset = load_from_disk(CACHE_PATH)
        except FileNotFoundError:
            print("Dataset not found, loading from huggingface")
            dataset = self.get_dataset(split=split, streaming=streaming)
            dataset = dataset.remove_columns("qid")
            dataset = dataset.rename_column("Question", "query")
            dataset = dataset.rename_column("Answer", "label")
            dataset = dataset.rename_column("Tweet", "document")
            dataset = dataset.map(self.preprocess)

            # dataset.save_to_disk(CACHE_PATH)  # TODO: unable to save IterableDataset
            self.dataset = dataset

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def get_dataset(self, split="train", streaming=False) -> dict:
        dataset = load_dataset(
            self.dataset_uuid, split=split, streaming=streaming
        ).with_format("torch")
        return dataset

    def preprocess(self, example: dict) -> dict:
        """Preprocess the dataset with basic cleanup and splitting into query and document"""
        # test split does not have a label
        example["label"] = None if not example["label"] else example["label"][0]
        example["document"] = tweet_preprocessor.clean(example["document"])

        return example

    def tokenize(self, example: dict) -> dict:
        """Tokenize the dataset with the encoder tokenizer"""
        tokenized_query = self.tokenizer(
            example["query"],
            padding=True,
            truncation=False,
            return_tensors=None,
            pad_to_multiple_of=8,
        )
        tokenized_document = self.tokenizer(
            example["document"],  # TODO: check if each chunk has a BOS token
            padding=True,
            truncation=True,
            max_length=self.chunk_size,
            return_tensors=None,
            pad_to_multiple_of=8,
            return_overflowing_tokens=True,
        )

        return_dict = {
            "query_ids": tokenized_query.input_ids,
            "query_attention_mask": tokenized_query.attention_mask,
            "document_ids": tokenized_document.input_ids,
            "document_attention_mask": tokenized_document.attention_mask,
        }

        if "label" in example and example["label"] is not None:
            tokenized_output = self.tokenizer(
                example["label"],
                padding=True,
                truncation=False,
                return_tensors=None,
                pad_to_multiple_of=8,
            )
            return_dict["label_ids"] = tokenized_output.input_ids
            return_dict["label_attention_mask"] = tokenized_output.attention_mask

        return return_dict


class MuLD_Dataset(HFDataset):
    """HuggingFace MuLD Dataset for NarrativeQA
    Downloads the dataset from HuggingFace or loads from disk
    Preprocesses - cleans BOM and HTML tags, extra spaces and new lines; splits into query and document
    Tokenizes the dataset with the given tokenizer; Document is chunked, query and output [optional] are not
    Keys in the example:
        `query`: str
        `document`: str
        `label` [optional]: str
        `query_ids`: torch.Tensor of shape (1, query_length)
        `query_attention_mask`: torch.Tensor of shape (1, query_length)
        `document_ids`: torch.Tensor of shape (1, num_chunks, chunk_size)
        `document_attention_mask`: torch.Tensor of shape (1, num_chunks, chunk_size)
        `label_ids`[optional]: torch.Tensor of shape (1, output_length)
        `label_attention_mask` [optional]: torch.Tensor of shape (1, output_length)
    """

    def __init__(
        self,
        tokenizer,
        split="train",
        chunk_size=4096,
        streaming=False,
        hdf5_file_path=None,
        **kwargs,
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
            return_tensors=None,
            pad_to_multiple_of=8,
        )
        tokenized_document = self.tokenizer(
            example["document"],  # TODO: check if each chunk has a BOS token
            padding=True,
            truncation=True,
            max_length=self.chunk_size,
            return_tensors=None,
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
                return_tensors=None,
                pad_to_multiple_of=8,
            )
            return_dict["label_ids"] = tokenized_output.input_ids
            return_dict["label_attention_mask"] = tokenized_output.attention_mask

        return return_dict


class SQuAD_Dataset(HFDataset):
    """HuggingFace SQuAD Dataset for Abstractive QA
    Downloads the dataset from HuggingFace or loads from disk
    Preprocesses - cleans BOM and HTML tags, extra spaces and new lines; splits into query and document
    Tokenizes the dataset with the given tokenizer; Does not chunk the document
    Keys in the example:
        `input`: str
        `label` [optional]: str
        `input_ids`: torch.Tensor of shape (1, input_length)
        `input_attention_mask`: torch.Tensor of shape (1, input_length)
        `labels`[optional]: torch.Tensor of shape (1, output_length)
        `label_attention_mask` [optional]: torch.Tensor of shape (1, output_length)
    """

    def __init__(
        self,
        tokenizer,
        split="train",
        streaming=False,
        mode="icl",
        debug=False,
        **kwargs,
    ):
        super(SQuAD_Dataset, self).__init__(dataset_uuid="squad", **kwargs)

        if mode != "icl":
            raise NotImplementedError("Only ICL mode is supported")

        self.tokenizer = tokenizer
        preprocess = self.preprocess_icl
        dataset = self.get_dataset(split=split, streaming=streaming)
        if debug:
            dataset["train"] = dataset["train"].select(range(100))
            dataset["validation"] = dataset["validation"].select(range(20))
        for key, val in dataset.items():
            dataset[key] = val.shuffle(seed=42)
        dataset = dataset.map(
            preprocess,
            batched=True,
            remove_columns=["id", "title", "context", "question", "answers"],
            load_from_cache_file=False,
        )

        self.dataset = dataset

    def get_dataset(self, split="train", streaming=False) -> dict:
        dataset = load_dataset(
            self.dataset_uuid, split=split, streaming=streaming
        ).with_format("torch")
        return dataset

    def preprocess_icl(self, example: dict) -> dict:
        """Preprocess the dataset with basic cleanup and splitting into query
        and document"""

        questions = [q.strip() for q in example["question"]]

        formatted_texts = []
        for context, question in zip(example["context"], questions):
            formatted_texts.append(f"Context: {context} Question: {question}")

        model_inputs = self.tokenizer(
            formatted_texts,
            max_length=384,
            truncation=True,
            return_offsets_mapping=False,
            padding="max_length",
        )

        # create global_attention_mask lists with 1 at the first token
        # of each question and 0 for the rest
        global_attention_masks = []
        for input_ids in model_inputs["input_ids"]:
            global_attention_masks.append([1] + [0] * (len(input_ids) - 1))
        model_inputs["global_attention_mask"] = global_attention_masks

        text_target = ["Answer: " + answer["text"][0] for answer in example["answers"]]
        labels = self.tokenizer(
            text_target=text_target,
            max_length=32,  # TODO: this is a guess
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class DataCollatorForLDQA:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str] = True,
        max_query_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
        hdf5_file_path: str = None,
        max_chunks_for_doc: int = 150,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_query_length = max_query_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.hdf5_file_path = hdf5_file_path
        self.max_chunks_for_doc = max_chunks_for_doc

    def __call__(self, encoded_inputs, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(
            encoded_inputs[0], dict
        ):
            encoded_inputs = {
                key: [example[key] for example in encoded_inputs]
                for key in encoded_inputs[0].keys()
            }

        tokenized_query = self.tokenizer(
            encoded_inputs["query"],
            padding=True,
            truncation=False,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )
        query_ids = tokenized_query.input_ids
        query_attention_mask = tokenized_query.attention_mask
        batch = {
            "query_ids": query_ids,
            "query_attention_mask": query_attention_mask,
        }

        document_outputs = torch.zeros(1)
        document_ids = torch.zeros(1)
        document_attention_mask = torch.zeros(1)

        tokenized_document = self.tokenizer(
            encoded_inputs["document"],  # TODO: check if each chunk has a BOS token
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
            pad_to_multiple_of=8,
            return_overflowing_tokens=True,
        )
        document_ids = tokenized_document.input_ids
        document_attention_mask = tokenized_document.attention_mask
        # tokenizer returns a list of shape ~(batch_size * num_chunks, chunk_size)
        # reshape tokenized_document to (batch_size, max_num_chunks, chunk_size)
        # using overflow_to_sample_mapping
        # Note that num_chunks is not constant for all samples in the batch
        document_ids, document_attention_mask = self._reshape_tokenized_document(
            document_ids,
            document_attention_mask,
            tokenized_document.overflow_to_sample_mapping,
        )
        batch["document_ids"] = document_ids
        batch["document_attention_mask"] = document_attention_mask

        if self.hdf5_file_path:
            document_encoding_outputs = self._prepare_hdf5_outputs(encoded_inputs)
            batch["document_encoding_outputs"] = document_encoding_outputs

        # expected shapes
        # document_ids: (batch_size, max_num_chunks, chunk_size)
        # document_attention_mask: (batch_size, max_num_chunks, chunk_size)
        # document_encoding_outputs: (batch_size, max_num_chunks, chunk_size, hidden_size)

        if "label" in encoded_inputs:
            tokenized_output = self.tokenizer(
                encoded_inputs["label"],
                padding=True,
                truncation=False,
                return_tensors="pt",
                pad_to_multiple_of=8,
            )
            label_ids = tokenized_output.input_ids
            # ignore the padding tokens
            label_ids[label_ids == self.tokenizer.pad_token_id] = -100
            batch["labels"] = label_ids
        return batch

    def _prepare_hdf5_outputs(self, encoded_inputs: Dict) -> torch.Tensor:
        """Given a batch of encoded inputs, loads their embeddings from the H5 file and returns
        Pads the embeddings appropriately as described below and return their attention masks
        1. If the document has only one chunk, pad the batch to max sequence length in batch.
        2. If the document has multiple chunks, pad the batch to max_chunks_for_doc chunks.

        Args:
        :param encoded_inputs: Dict containing the encoded inputs

        Returns:
        :return document_outputs: torch.Tensor of shape
        [batch_size, max_num_chunks, chunk_size, hidden_size] or
        [batch_size, 1, max_length_in_batch, hidden_size]
        """
        with h5py.File(self.hdf5_file_path, "r", swmr=True) as f:
            all_document_outputs = []
            for docs in encoded_inputs["document"]:
                key = sha256(docs.encode("utf-8")).hexdigest()
                doc_output = torch.Tensor(f[key][...])
                all_document_outputs.append(doc_output)
            all_document_outputs = self._reshape_encoded_documents(all_document_outputs)

            return all_document_outputs

    def _reshape_encoded_documents(
        self, document_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Reshapes batch of encoded documents to (batch_size, max_num_chunks, chunk_size, hidden_size)
        if the document has multiple chunks else to
        (batch_size, 1, max_length_in_batch, hidden_size)

        Args:
        :param document_outputs: List of torch.Tensor of shape
        [num_chunks, context_length, hidden_size]

        Returns:
        :return reshaped_encodings: torch.Tensor of shape
        [batch_size, max_num_chunks, chunk_size, hidden_size]
        :return attention_mask: torch.Tensor of shape
        [batch_size, max_num_chunks, chunk_size]
        """

        # short document, pad to max sequence length in batch
        if self.max_chunks_for_doc == 1:
            document_outputs = [x.squeeze(0) for x in document_outputs]
            document_outputs = torch.nn.utils.rnn.pad_sequence(
                document_outputs,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )

            document_outputs = document_outputs.unsqueeze(1)
            return document_outputs

        # long document, pad to max_chunk_size
        else:
            padded_outputs = [self._pad_to_max_chunks(x) for x in document_outputs]
            document_outputs = torch.stack(padded_outputs, dim=0)
            return document_outputs

    def _pad_to_max_chunks(self, document_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Pads a single document embedding to max_chunks_for_doc chunks"""
        num_chunks = document_outputs.shape[0]
        max_chunks_in_batch = max(x.shape[0] for x in document_outputs)
        # change size of document_outputs to (batch_size, max_chunks, context_length, hidden_size)
        # by adding zeros in that place
        document_outputs = F.pad(
            document_outputs,
            (0, 0, 0, 0, 0, max_chunks_in_batch - num_chunks),
            value=self.tokenizer.pad_token_id,
        )

        return document_outputs

    def _reshape_tokenized_document(
        self,
        document_ids: torch.Tensor,
        document_attention_mask: torch.Tensor,
        overflow_to_sample_mapping: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshape the tokenized document to (batch_size, max_num_chunks, chunk_size)
        where max_num_chunks is the maximum number of chunks for a document in the batch.
        `document_ids` and `document_attention_mask` are of shape ~(batch_size * num_chunks, chunk_size)
        `overflow_to_sample_mapping` is of shape ~(batch_size * num_chunks)
        where `num_chunks` could be different for each document in the batch
        """

        batch_size = overflow_to_sample_mapping.max().item() + 1
        chunk_size = document_ids.shape[-1]
        max_num_chunks = overflow_to_sample_mapping.bincount().max().item()

        # reshape document_ids and document_attention_mask to (batch_size, max_num_chunks, chunk_size)
        reshaped_document_ids = torch.zeros(
            (batch_size, max_num_chunks, chunk_size), dtype=document_ids.dtype
        )
        reshaped_document_attention_mask = torch.zeros(
            (batch_size, max_num_chunks, chunk_size),
            dtype=document_attention_mask.dtype,
        )

        # iterate over each sample in the batch
        for i in range(batch_size):
            # get the indices of the chunks for the current sample
            sample_indices = (overflow_to_sample_mapping == i).nonzero(as_tuple=True)[0]
            num_chunks = sample_indices.shape[0]

            # get the corresponding document_ids and document_attention_mask
            sample_document_ids = document_ids[sample_indices]
            sample_document_attention_mask = document_attention_mask[sample_indices]

            # pad the document_ids and document_attention_mask to max_num_chunks
            sample_document_ids = F.pad(
                sample_document_ids,
                (0, 0, 0, max_num_chunks - num_chunks),
                value=self.tokenizer.pad_token_id,
            )
            sample_document_attention_mask = F.pad(
                sample_document_attention_mask,
                (0, 0, 0, max_num_chunks - num_chunks),
                value=0,
            )

            # add the sample to the batch
            reshaped_document_ids[i] = sample_document_ids
            reshaped_document_attention_mask[i] = sample_document_attention_mask

        return reshaped_document_ids, reshaped_document_attention_mask


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    muld = MuLD_Dataset(streaming=True, split=None, tokenizer=tokenizer)
    for ex in muld.dataset["train"]:
        break
    print(f"{ex.keys()=}")
    print(f"{ex['query']=}")
    print(f"{ex['query_ids'].shape=}")
    print(f"{ex['document_ids'].shape=}")
    print(f"{ex['query_attention_mask'].shape=}")
    print(f"{ex['document_attention_mask'].shape=}")
