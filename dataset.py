from datasets import load_dataset


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

    def get_dataset(self) -> dict:
        dataset = load_dataset(self.dataset_uuid, 'NarrativeQA')
        return dataset
    
if __name__ == '__main__':
    muld = MuLD_Dataset()
    print(muld.get_dataset())
