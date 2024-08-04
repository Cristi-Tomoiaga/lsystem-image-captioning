import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

from first_cnnlstm_model.vocabulary import Vocabulary
from first_cnnlstm_model.metrics import check_lword_syntax
from dataset_generator.lword_renderer import LWordRenderer
from dataset_generator.lword_preprocessor import LWordPreprocessor


class LSystemDataset(Dataset):
    def __init__(self, dataset_type, dataset_version, root_dir, vocabulary: Vocabulary, transform=None):
        path = root_dir

        if dataset_type == 'train':
            path = os.path.join(path, 'train')
        elif dataset_type == 'valid':
            path = os.path.join(path, 'valid')
        elif dataset_type == 'test':
            path = os.path.join(path, 'test')

        if dataset_version == 2:
            self.__epoch_data = pd.read_csv(os.path.join(root_dir, 'epoch_data.csv'), header=None, names=['angle', 'distance'])
            self.__current_epoch = 0
            self.__renderer = LWordRenderer(512, 512)

        self.__dataset_version = dataset_version
        self.__root_dir = path
        self.__vocabulary = vocabulary
        self.__transform = transform

        if dataset_version == 1:
            self.__captions = pd.read_csv(os.path.join(path, 'captions.csv'), header=None, names=['lword', 'image', 'angle', 'distance'])
        elif dataset_version == 2:
            self.__captions = pd.read_csv(os.path.join(path, 'captions.csv'), header=None, names=['lword'])

    def __len__(self):
        return len(self.__captions)

    def __prepare_lword(self, lword, angle, distance):
        converted_lword = ''.join(self.__vocabulary.tokenizer(lword))

        while converted_lword != lword:
            lword = converted_lword

            if not check_lword_syntax(converted_lword, angle, distance, strict=True):
                converted_lword = self.__renderer.fix_lword_geometrically(converted_lword, angle, distance)
                converted_lword = LWordPreprocessor.process_lword_repeatedly(converted_lword)

            converted_lword = ''.join(self.__vocabulary.tokenizer(converted_lword))

        return converted_lword

    def __getitem__(self, idx):
        if self.__dataset_version == 1:  # This dataset lacks the extra preprocessing from the second version
            lword = self.__captions.iloc[idx, 0]
            image_path = self.__captions.iloc[idx, 1]
            angle = self.__captions.iloc[idx, 2]
            distance = self.__captions.iloc[idx, 3]

            image = Image.open(os.path.join(self.__root_dir, image_path))
            if self.__transform is not None:
                image = self.__transform(image)

            target = self.__vocabulary.convert_from_lword(lword)

            return image, torch.Tensor(target), torch.Tensor([angle]), torch.Tensor([distance])
        elif self.__dataset_version == 2:
            lword = self.__captions.iloc[idx, 0]
            angle = self.__epoch_data.iloc[self.__current_epoch, 0]
            distance = self.__epoch_data.iloc[self.__current_epoch, 1]

            lword = self.__renderer.fix_lword_geometrically(lword, angle, distance)
            lword = LWordPreprocessor.process_lword_repeatedly(lword)
            lword = self.__prepare_lword(lword, angle, distance)

            image = self.__renderer.render(lword, angle, distance, rescale=True)
            if self.__transform is not None:
                image = self.__transform(image)

            target = self.__vocabulary.convert_from_lword(lword)

            return image, torch.Tensor(target), torch.Tensor([angle]), torch.Tensor([distance])

    def set_epoch(self, epoch):
        if self.__dataset_version == 2:
            self.__current_epoch = epoch
