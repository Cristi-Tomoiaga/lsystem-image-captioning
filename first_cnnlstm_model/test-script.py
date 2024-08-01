# import torch.nn.utils.rnn
# import numpy as np
# from matplotlib import pyplot as plt
from torchvision import transforms

import first_cnnlstm_model.lsystem_dataloaders as lsystem_dataloaders
# from first_cnnlstm_model.lsystem_dataset import LSystemDataset
from first_cnnlstm_model.vocabulary import Vocabulary
from first_cnnlstm_model.model import EncoderCNN, DecoderRNN
import first_cnnlstm_model.utils as utils

transform = transforms.Compose([
    transforms.ToTensor(),
])

vocabulary = Vocabulary()

dataloader_v1 = lsystem_dataloaders.get_train_loader(
    root_dir='../generated_datasets/lsystem_dataset_48267__01_08_2024_15_33',
    version=1,
    transform=transform,
    vocabulary=vocabulary,
    batch_size=128,
    num_workers=4
)
dataloader_v2 = lsystem_dataloaders.get_train_loader(
    root_dir='../generated_datasets/lsystem_dataset_v2_48267__01_08_2024_15_55',
    version=2,
    transform=transform,
    vocabulary=vocabulary,
    batch_size=128,
    num_workers=4
)

# mean, std = utils.compute_mean_std_for_dataset("train", 1, "../generated_datasets/lsystem_dataset_48267__01_08_2024_15_33")
# max_sequence_length = utils.compute_max_sequence_length_for_dataset("train", 1, "../generated_datasets/lsystem_dataset_48267__01_08_2024_15_33")
#
# print("Mean, std: ", mean, std)
# print("Mean ", mean.item())
# print("Std ", std.item())
# print("Max sequence length ", max_sequence_length)

# mean, std = utils.compute_mean_std_for_dataset("train", 2, "../generated_datasets/lsystem_dataset_v2_48267__01_08_2024_15_55")
# max_sequence_length = utils.compute_max_sequence_length_for_dataset("train", 2, "../generated_datasets/lsystem_dataset_v2_48267__01_08_2024_15_55", 5)
#
# print("Mean, std: ", mean, std)
# print("Mean ", mean.item())
# print("Std ", std.item())
# print("Max sequence length ", max_sequence_length)

# for epoch in range(495):
#     dataloader_v2.dataset.set_epoch(epoch)
#
#     for i, (images, targets, lengths, angles, distances) in enumerate(dataloader_v2):
#         print(i, "->", images.shape, targets.shape, len(lengths), angles.shape, distances.shape)
#
#     print(epoch)

# dataset_v2 = LSystemDataset("train", 1, "../generated_datasets/lsystem_dataset_v2_48267__01_08_2024_15_55", vocabulary, transform)
# for epoch in range(495):
#     dataset_v2.set_epoch(epoch)
#
#     for image, target, angle, distance in dataset_v2:
#         pass
#
#     print(epoch)

print(len(dataloader_v1), len(vocabulary))

images, targets, lengths, angles, distances = next(iter(dataloader_v1))
print(images.shape, targets.shape, len(lengths), angles.shape, distances.shape)
print(targets[-1])
print(vocabulary.convert_to_lword(targets[-1].numpy()))

print(len(dataloader_v2), len(vocabulary))

images, targets, lengths, angles, distances = next(iter(dataloader_v2))
print(images.shape, targets.shape, len(lengths), angles.shape, distances.shape)
print(targets[-1])
print(vocabulary.convert_to_lword(targets[-1].numpy()))

# for i, (images, targets, lengths, angles, distances) in enumerate(dataloader_v1):
#     converted_targets = [vocabulary.convert_to_lword(target.squeeze(0).numpy()) for target in targets.split(1)]
#     print(i, "->", converted_targets, lengths, angles, distances)
#
#     imgs = [img.squeeze() for img in images.split(1)]
#     _, axs = plt.subplots(4, 3, figsize=(12, 12))
#     axs = axs.flatten()
#     for img, ax in zip(imgs, axs):
#         ax.imshow(np.asarray(img), cmap="gray")
#     plt.show()
#
# print()
#
# for epoch in range(5):
#     dataloader_v2.dataset.set_epoch(epoch)
#
#     for i, (images, targets, lengths, angles, distances) in enumerate(dataloader_v2):
#         converted_targets = [vocabulary.convert_to_lword(target.squeeze(0).numpy()) for target in targets.split(1)]
#         print(i, "->", converted_targets, lengths, angles, distances)
#
#         imgs = [img.squeeze() for img in images.split(1)]
#         _, axs = plt.subplots(4, 3, figsize=(12, 12))
#         axs = axs.flatten()
#         for img, ax in zip(imgs, axs):
#             ax.imshow(np.asarray(img), cmap="gray")
#         plt.show()

# plt.imshow(images[0].squeeze(), cmap='gray')
# plt.show()
# print(targets[0])
#
# print(targets)
#
# pack = torch.nn.utils.rnn.pack_padded_sequence(targets, lengths, batch_first=True)
# print(pack)
#
# embed = torch.nn.Embedding(len(vocabulary), 10)
# print(embed(pack[0]))

encoder = EncoderCNN(feature_size=128)
decoder = DecoderRNN(embed_size=128, hidden_size=256, vocab_size=len(vocabulary))

utils.count_parameters(encoder)
utils.count_parameters(decoder)
