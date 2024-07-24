import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

from vocabulary import Vocabulary


class LSystemDataset(Dataset):
    def __init__(self, dataset_type, root_dir, vocabulary: Vocabulary, transform=None):
        path = root_dir

        if dataset_type == 'train':
            path = os.path.join(path, 'train')
        elif dataset_type == 'valid':
            path = os.path.join(path, 'valid')
        elif dataset_type == 'test':
            path = os.path.join(path, 'test')

        self.__root_dir = path
        self.__vocabulary = vocabulary
        self.__transform = transform
        self.__captions = pd.read_csv(os.path.join(path, 'captions.csv'), header=None, names=['lword', 'image'])

    def __len__(self):
        return len(self.__captions)

    def __getitem__(self, idx):
        lword = self.__captions.iloc[idx, 0]
        image_path = self.__captions.iloc[idx, 1]

        image = Image.open(os.path.join(self.__root_dir, image_path))
        if self.__transform is not None:
            image = self.__transform(image)

        target = self.__vocabulary.convert_lword(lword)

        return image, torch.Tensor(target)
