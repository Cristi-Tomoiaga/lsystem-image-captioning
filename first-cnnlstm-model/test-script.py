from torchvision import transforms
import matplotlib.pyplot as plt

import lsystem_dataloaders

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataloader = lsystem_dataloaders.get_train_loader(
    root_dir='../generated_datasets/lsystem_dataset_20__23_07_2024_23_37',
    transform=transform,
    batch_size=32,
    num_workers=4
)

print(len(dataloader))

images, targets, lengths = next(iter(dataloader))
print(images.shape, targets.shape, len(lengths))

plt.imshow(images[0].squeeze(), cmap='gray')
plt.show()
print(targets[0])
