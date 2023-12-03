from dataset import MuLD_Dataset, TweetQA_Dataset

DATASET_CONFIG = {
    "TweetQA": {
        "cls": TweetQA_Dataset,
        "max_chunks_for_doc": 1,
        "hdf5_path": "/coc/flash8/akutumbaka3/DDRL/data/tweetqa_full_global/embeddings.h5",
    },
    "MuLD": {
        "cls": MuLD_Dataset,
        "max_chunks_for_doc": 150,
        "hdf5_path": "/coc/flash8/akutumbaka3/LDQA-with-Adapters/data/longformer_noglobal/embeddings.h5",
    },
}
