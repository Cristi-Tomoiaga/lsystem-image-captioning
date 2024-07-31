import torch
from torch.utils.data import DataLoader

from first_cnnlstm_model.lsystem_dataset import LSystemDataset


class LWordCollate:
    def __init__(self, pad_idx):
        self.__pad_idx = pad_idx

    def __call__(self, batch):
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        images, targets = zip(*batch)

        images = torch.stack(images, 0)  # (batch_size, 1, 512, 512)

        lengths = [len(target) for target in targets]  # (batch_size)
        transformed_targets = torch.ones(len(targets), max(lengths)).long() * self.__pad_idx  # (batch_size, max_target_length)

        for i, target in enumerate(targets):
            end = lengths[i]
            transformed_targets[i, :end] = target[:end]

        return images, transformed_targets, lengths


def get_train_loader(root_dir, transform, vocabulary, batch_size, num_workers, shuffle=True):
    dataset = LSystemDataset('train', root_dir, vocabulary, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=LWordCollate(vocabulary("<pad>"))
    )


def get_valid_loader(root_dir, transform, vocabulary, batch_size, num_workers, shuffle=False):
    dataset = LSystemDataset('valid', root_dir, vocabulary, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=LWordCollate(vocabulary("<pad>"))
    )


def get_test_loader(root_dir, transform, vocabulary, batch_size, num_workers, shuffle=False):
    dataset = LSystemDataset('test', root_dir, vocabulary, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=LWordCollate(vocabulary("<pad>"))
    )
