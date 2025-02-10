from datasets import load_dataset

def load_imdb_dataset():
    dataset = load_dataset("stanfordnlp/imdb")
    return dataset