import torch
from torch.utils.data import DataLoader

from lsystem_dataset import LSystemDataset


class LWordCollate:
    def __call__(self, batch):
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        images, targets = zip(*batch)

        images = torch.stack(images, 0)  # (batch_size, 1, 512, 512)

        lengths = [len(target) for target in targets]  # (batch_size)
        transformed_targets = torch.zeros(len(targets), max(lengths)).long()  # (batch_size, max_target_length), padding=0

        for i, target in enumerate(targets):
            end = lengths[i]
            transformed_targets[i, :end] = target[:end]

        return images, transformed_targets, lengths


def get_train_loader(root_dir, transform, batch_size, num_workers, shuffle=True):
    dataset = LSystemDataset('train', root_dir, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=LWordCollate()
    )


def get_valid_loader(root_dir, transform, batch_size, num_workers, shuffle=False):
    dataset = LSystemDataset('valid', root_dir, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=LWordCollate()
    )


def get_test_loader(root_dir, transform, batch_size, num_workers, shuffle=False):
    dataset = LSystemDataset('test', root_dir, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=LWordCollate()
    )
