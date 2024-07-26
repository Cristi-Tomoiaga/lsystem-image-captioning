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
    root_dir='../generated_datasets/lsystem_dataset_20__23_07_2024_23_37',
    transform=transform,
    vocabulary=vocabulary,
    batch_size=32,
    num_workers=4
)

print(utils.compute_mean_std_for_dataset("train", "../generated_datasets/lsystem_dataset_20__23_07_2024_23_37"))
print(utils.compute_max_sequence_length_for_dataset("train", "../generated_datasets/lsystem_dataset_20__23_07_2024_23_37"))

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
decoder = DecoderRNN(embed_size=256, hidden_size=128, vocab_size=len(vocabulary), max_sequence_length=50)

utils.count_parameters(encoder)
utils.count_parameters(decoder)
