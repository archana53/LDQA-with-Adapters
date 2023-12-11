from dataset import MuLD_Dataset, TweetQA_Dataset, SQuAD_Dataset

DATASET_CONFIG = {
    "TweetQA": {
        "cls": TweetQA_Dataset,
        "max_chunks_for_doc": 1,
        "hdf5_path": "/coc/flash8/akutumbaka3/DDRL/data/tweetqa_full_global/embeddings.h5",
        "streaming": False,
        "generation_config": {
            "max_length": 16,
        },
    },
    "MuLD": {
        "cls": MuLD_Dataset,
        "max_chunks_for_doc": 142,
        "hdf5_path": "/coc/flash8/akutumbaka3/LDQA-with-Adapters/data/longformer_noglobal/embeddings.h5",
        "streaming": True,
    },
    "SQuAD": {
        "cls": SQuAD_Dataset,
        "max_chunks_for_doc": 1,
        "hdf5_path": "/coc/flash8/akutumbaka3/DDRL/data/squad_full_global/embeddings.h5",
        "streaming": False,
        "generation_config": {
            "max_length": 30,
        },
    },
}
