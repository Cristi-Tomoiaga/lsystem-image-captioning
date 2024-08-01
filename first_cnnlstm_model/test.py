import argparse

import numpy as np
import torch
from torchvision import transforms
from torch import nn

import first_cnnlstm_model.utils as utils
from first_cnnlstm_model.vocabulary import Vocabulary
from first_cnnlstm_model.model import EncoderCNN, DecoderRNN
from first_cnnlstm_model.lsystem_dataloaders import get_test_loader
from first_cnnlstm_model.metrics import AverageMetric


def test(args):
    device = utils.get_device()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((args.mean,), (args.std,))
    ])

    vocab = Vocabulary()

    dataset_version = 1 if args.epochs == 0 else 2
    num_epochs = 1 if dataset_version == 1 else args.epochs

    test_dataloader = get_test_loader(
        root_dir=args.dataset_path,
        version=dataset_version,
        transform=transform,
        vocabulary=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    encoder = EncoderCNN(feature_size=args.embed_size)
    decoder = DecoderRNN(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab)
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    checkpoint = utils.load_checkpoint(args.model_path, move_to_cpu=False)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    test_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab("<pad>"))

    test_loss = AverageMetric()
    test_perplexity = AverageMetric()

    test_loss.reset()  # investigate balancing loss with batch size
    test_perplexity.reset()

    with torch.no_grad():
        for epoch in range(num_epochs):
            if dataset_version == 2:
                test_dataloader.dataset.set_epoch(epoch)

            for i, (images, captions, lengths, _, _) in enumerate(test_dataloader):
                images = images.to(device)
                captions = captions.to(device)
                max_sequence_length = captions.size()[-1]

                features = encoder(images)
                outputs = decoder.generate_caption(features, max_sequence_length, return_idx=False)

                loss = test_loss_fn(outputs.view(-1, outputs.size(dim=-1)), captions.view(-1))
                test_loss.add_value(loss.item())
                test_perplexity.add_value(np.exp(loss.item()))

    print(f"Test results:"
          f" Average Loss: {test_loss.average_value:.4f}, Average Perplexity: {test_perplexity.average_value:5.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../models/', help='The path to the saved model')
    parser.add_argument('--dataset_path', type=str, default='../generated_datasets/lsystem_dataset_v2_48267__01_08_2024_15_55', help='The path of the dataset')
    parser.add_argument('--epochs', type=int, default=495, help='The number of augmentation epochs')
    parser.add_argument('--mean', type=float, default=0.9947, help='The mean value of the dataset')
    parser.add_argument('--std', type=float, default=0.0730, help='The standard deviation of the dataset')

    # Model parameters (same as train.py)
    parser.add_argument('--embed_size', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden state dimension')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

    parsed_args = parser.parse_args()
    print(parsed_args)
    test(parsed_args)
