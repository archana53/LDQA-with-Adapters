from datasets import load_dataset
from torch import utils
from torch.utils.data import DataLoader


class Dataset(object):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__(**kwargs)

    def get_dataset(self) -> dict:
        """Returns the corresponding dataset from huggingface"""
        raise NotImplementedError


class MuLD_Dataset(Dataset):
    def __init__(self):
        super(MuLD_Dataset, self).__init__()
        self.dataset_uuid = "ghomasHudson/muld"

    def get_dataset(self, split="train", streaming=False) -> dict:
        dataset = load_dataset(
            self.dataset_uuid, "NarrativeQA", split=split, streaming=streaming
        )
        return dataset


if __name__ == "__main__":
    muld = MuLD_Dataset()
    dset = muld.get_dataset(split="validation", streaming=True)
    print(dset)
    train_dset = dset.with_format("torch")
    train_loader = DataLoader(train_dset, batch_size=4)
    # print(train_loader)
    for example in dset:
        # print(example.keys())
        # print(example["input"][:500])
        # print("output is ", example["output"])
        # print(example["metadata"])
        # dict containing 'input', 'output', 'metadata'
        break
