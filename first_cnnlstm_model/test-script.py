# import torch.nn.utils.rnn
from torchvision import transforms
# import matplotlib.pyplot as plt

import lsystem_dataloaders
from vocabulary import Vocabulary
from model import EncoderCNN, DecoderRNN
import utils

transform = transforms.Compose([
    transforms.ToTensor(),
])

vocabulary = Vocabulary()

dataloader = lsystem_dataloaders.get_train_loader(
    root_dir='../generated_datasets/lsystem_dataset_48267__30_07_2024_12_54',
    transform=transform,
    vocabulary=vocabulary,
    batch_size=32,
    num_workers=4
)

# mean, std = utils.compute_mean_std_for_dataset("train", "../generated_datasets/lsystem_dataset_48267__30_07_2024_12_54")
# max_sequence_length = utils.compute_max_sequence_length_for_dataset("train", "../generated_datasets/lsystem_dataset_48267__30_07_2024_12_54")
#
# print("Mean, std: ", mean, std)
# print("Mean ", mean.item())
# print("Std ", std.item())
# print("Max sequence length ", max_sequence_length)

print(len(dataloader), len(vocabulary))

images, targets, lengths = next(iter(dataloader))
print(images.shape, targets.shape, len(lengths))

print(targets[-1])
print(vocabulary.convert_to_lword(targets[-1].numpy()))

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
