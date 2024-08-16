# import torch.nn.utils.rnn
# import numpy as np
# import math

# import matplotlib.pyplot as plt
# import torch
# from PIL import Image
from torchvision import transforms
# import torch.nn.utils.rnn as rnn_utils

# import second_cnnlstm_model.lsystem_dataloaders as lsystem_dataloaders
from second_cnnlstm_model.lsystem_dataset import LSystemDataset
from second_cnnlstm_model.vocabulary import Vocabulary, LWordTokenizer
from second_cnnlstm_model.model import EncoderCNN, DecoderRNN
import second_cnnlstm_model.utils as utils
import second_cnnlstm_model.metrics as metrics
from dataset_generator.lword_renderer import LWordRenderer
from dataset_generator.lword_preprocessor import LWordPreprocessor

transform = transforms.Compose([
    transforms.ToTensor(),
])

vocabulary = Vocabulary()

# dataloader_v1 = lsystem_dataloaders.get_train_loader(
#     root_dir='../generated_datasets/lsystem_dataset_48267__01_08_2024_15_33',
#     version=1,
#     transform=transform,
#     vocabulary=vocabulary,
#     batch_size=128,
#     num_workers=4
# )
# dataloader_v2 = lsystem_dataloaders.get_train_loader(
#     root_dir='../generated_datasets/lsystem_dataset_v2_48267__01_08_2024_15_55',
#     version=2,
#     transform=transform,
#     vocabulary=vocabulary,
#     batch_size=128,
#     num_workers=4
# )

# mean, std = utils.compute_mean_std_for_dataset("train", 1, "../generated_datasets/lsystem_dataset_48267__01_08_2024_15_33")
# max_sequence_length = utils.compute_max_sequence_length_for_dataset("train", 1, "../generated_datasets/lsystem_dataset_48267__01_08_2024_15_33")
#
# print("Mean, std: ", mean, std)
# print("Mean ", mean.item())
# print("Std ", std.item())
# print("Max sequence length ", max_sequence_length)

# mean, std = utils.compute_mean_std_for_dataset("train", 2, "../generated_datasets/lsystem_dataset_v2_48267__01_08_2024_15_55")
# max_sequence_length = utils.compute_max_sequence_length_for_dataset("train", 2, "../generated_datasets/lsystem_dataset_v2_48267__01_08_2024_15_55", 1)
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

# print(len(dataloader_v1), len(vocabulary))
#
# images, targets, lengths, angles, distances = next(iter(dataloader_v1))
# print(images.shape, targets.shape, len(lengths), angles.shape, distances.shape)
# print(targets[-1])
# print(vocabulary.convert_to_lword(targets[-1].numpy()))
#
# print(len(dataloader_v2), len(vocabulary))
#
# images, targets, lengths, angles, distances = next(iter(dataloader_v2))
# print(images.shape, targets.shape, len(lengths), angles.shape, distances.shape)
# print(targets[-1])
# print(vocabulary.convert_to_lword(targets[-1].numpy()))

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


# sequences = ["F-F+F[+F-F[F]]", "F+F[F-F]"]
# tokens = [torch.tensor(vocabulary.convert_from_lword(sen)) for sen in sequences]
# lens = [len(sen) for sen in tokens]
#
# print(sequences)
# print(tokens, lens)
#
# padded = rnn_utils.pad_sequence(tokens, batch_first=True, padding_value=vocabulary('<pad>'))
# print(padded)
#
# # noinspection PyTypeChecker
# packed = rnn_utils.pack_padded_sequence(padded, lens, batch_first=True)
# print(packed)
#
# print(metrics.convert_padded_sequence(padded, vocabulary('<eos>'), vocabulary=vocabulary))
# print(metrics.convert_packed_padded_sequence(packed[0], lens, vocabulary=vocabulary))
#
#
# image1 = Image.open('../dataset_generator/test4.png')
# image2 = Image.open('../dataset_generator/test4_copy.png').convert('L')
#
# plt.imshow(image1, cmap='gray')
# plt.show()
# plt.imshow(image2, cmap='gray')
# plt.show()
#
# hd = metrics.compute_hausdorff_distance(image1, image2)
# print(hd)
# print(hd / math.sqrt(2 * 512 * 512) * 100)
# print(math.sqrt(2 * 512 * 512))  # image diagonal length
#
# image1.close()
# image2.close()

lword = '<bos>F[-F+F+F[-F-F][+F-F]][+F-F[-F[-F+F][+F]][+F-F[-F][+F]]]<eos>'
angle = 31.97050666809082
distance = 100.0
renderer = LWordRenderer(512, 512)
lword = lword.replace('<bos>', '').replace('<eos>', '')
print(metrics.check_lword_syntax(lword, angle, distance, strict=True))

print(lword)
lword = renderer.fix_lword_geometrically(lword.replace('<bos>', '').replace('<eos>', ''), angle, distance)
print(lword)
lword = LWordPreprocessor.process_lword_repeatedly(lword)
print(lword)

print(metrics.check_lword_syntax(lword, angle, distance, strict=True))
# image = renderer.render(lword, angle, distance, rescale=True)
# image.save('test.png')

# image = renderer.render('F-F[-F[-F[-F[-F-F+F][+F+F-F]][+F+F-F[-F][+F]]][+F+F-F-F+F]][+F[-F-F][+F-F[-F[-F-F][+F+F]][+F+F-F]]]', angle, distance, rescale=True)
# image.save('test1.png')
# image = renderer.render('F-F[-F[-F[-F[-F-F+F][+F+F-F]][+F+F-F[-F][+F]]][+F+F-F-F+F]][+F[-F-F++F[-F][+F]][+F-F[-F[-F-F][+F+F]][+F+F-F]]]', angle, distance, rescale=True)
# image.save('test2.png')
#
# image = renderer.render('F[-F+F+F-F-F+F+F][+F-F-F-F[-F-F[-F][+F]][+F[-F-F][+F+F]]]', angle, distance, rescale=True)
# image.save('test3.png')
# image = renderer.render('F[-F+F+F-F-F+F+F][+F-F-F[-F[-F-F[-F][+F]][+F[-F-F][+F+F]]][+[-F][+F[-F-F][+F+F]]]]', angle, distance, rescale=True)
# image.save('test4.png')

dataset_v2 = LSystemDataset("test", 2, "../generated_datasets/lsystem_dataset_v2_48267__01_08_2024_15_55", vocabulary, transform)
for epoch in range(1):
    dataset_v2.set_epoch(epoch)

    for image, target, angle, distance in dataset_v2:
        lword = vocabulary.convert_to_lword(target.long().numpy())
        lword = lword.replace('<bos>', '').replace('<eos>', '')
        result = metrics.check_lword_syntax(lword, angle, distance, strict=True)

        if not result:
            print('Entered')
            while not metrics.check_lword_syntax(lword, angle, distance, strict=True):
                lword = renderer.fix_lword_geometrically(lword, angle, distance)
                lword = LWordPreprocessor.process_lword_repeatedly(lword)

            result2 = metrics.check_lword_syntax(lword, angle, distance, strict=True)
            if not result2:
                print('Not')
            else:
                print('Fixed')

    print(epoch)


lword = '<bos>F[-F+F+F[-F-F][+F-F]][+F-F[-F[-F+F+][++F]][+F-F-[[-F][+F]]]]<eos>'.replace('<bos>', '').replace('<eos>', '')
print(lword)

tokens = vocabulary.convert_from_lword(lword)
converted_lword = vocabulary.convert_to_lword(tokens).replace('<bos>', '').replace('<eos>', '')
print(converted_lword)

tokenizer = LWordTokenizer()
tokens = tokenizer(lword)
converted_lword = ''.join(tokens)
print(converted_lword)
