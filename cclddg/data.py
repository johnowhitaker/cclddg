# AUTOGENERATED! DO NOT EDIT! File to edit: 02_Datasets.ipynb (unless otherwise specified).

__all__ = ['CelebAData', 'CC12M']

# Cell
from datasets import load_dataset
class CelebAData():
    def __init__(self, max_examples=-1, batch_size=32, shuffle=False):

        # We'll always stick to this - might mean leaving behind a few examples
        self.batch_size = batch_size

        # Downloads ~1.3GB even if you only want a few examples - sorry!
        self.dataset = load_dataset('huggan/CelebA-faces')['train']

        # Optionally restrict to the first max_examples examples
        if max_examples>0:
            self.dataset = self.dataset.select(range(max_examples))

        if shuffle:
            self.dataset = self.dataset.shuffle()

        self.idx = 0 # Keep track of where we are in the data

    def get_batch(self):
        if (self.idx + self.batch_size)>len(self.dataset):
            self.idx = 0
        if self.batch_size>len(self.dataset):
            return []
        batch = self.dataset[self.idx:self.idx+self.batch_size]
        self.idx += self.batch_size

        images = [x for x in batch['image']]
        captions = ['a photo of a face' for _ in batch['image']]

        return images, captions

    def reset(self):
        self.idx = 0


# Cell
import webdataset as wds
class CC12M():
    def __init__(self, batch_size=32):

        self.batch_size=batch_size
        self.urls = 'https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01200}.tar'
        self.dataset = (wds.WebDataset(self.urls, handler=wds.warn_and_continue)
                   .shuffle(2500)
                   .decode("pil")
                   .map(self.preprocess)
                   .to_tuple("image", "text")
                   .batched(batch_size))

    def preprocess(self, sample):
            return {"image": sample['jpg'],
                    "text": sample['txt']}

    def get_batch(self):
        images, captions = next(iter(self.dataset))
        return images, captions

    def reset(self):
       self. dataset = (wds.WebDataset(self.urls, handler=wds.warn_and_continue)
                   .shuffle(2500)
                   .decode("pil")
                   .map(self.preprocess)
                   .to_tuple("image", "text")
                   .batched(self.batch_size))