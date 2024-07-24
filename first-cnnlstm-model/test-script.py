import torch.nn.utils.rnn
from torchvision import transforms
import matplotlib.pyplot as plt

import lsystem_dataloaders
from vocabulary import Vocabulary

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

print(len(dataloader), len(vocabulary))

images, targets, lengths = next(iter(dataloader))
print(images.shape, targets.shape, len(lengths))

plt.imshow(images[0].squeeze(), cmap='gray')
plt.show()
print(targets[0])

print(targets)

pack = torch.nn.utils.rnn.pack_padded_sequence(targets, lengths, batch_first=True)
print(pack)

embed = torch.nn.Embedding(len(vocabulary), 10)
print(embed(pack[0]))
