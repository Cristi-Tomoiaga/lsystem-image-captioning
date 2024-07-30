import argparse

import numpy as np
import torch
from torchvision import transforms
from torch import nn

import utils
from vocabulary import Vocabulary
from model import EncoderCNN, DecoderRNN
from lsystem_dataloaders import get_test_loader


def test(args):
    device = utils.get_device()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((args.mean,), (args.std,))
    ])

    vocab = Vocabulary()

    test_dataloader = get_test_loader(
        root_dir=args.dataset_path,
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

    total_test_batches = len(test_dataloader)
    running_test_loss = 0.0
    running_test_perplexity = 0.0  # investigate balancing loss with batch size
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(test_dataloader):
            images = images.to(device)
            captions = captions.to(device)
            max_sequence_length = captions.size()[-1]

            features = encoder(images)
            outputs = decoder.generate_caption(features, max_sequence_length, return_idx=False)

            loss = test_loss_fn(outputs.view(-1, outputs.size(dim=-1)), captions.view(-1))
            running_test_loss += loss.item()
            running_test_perplexity += np.exp(loss.item())

    avg_test_loss = running_test_loss / total_test_batches
    avg_test_perplexity = running_test_perplexity / total_test_batches

    print(f"Test results:"
          f" Average Loss: {avg_test_loss:.4f}, Average Perplexity: {avg_test_perplexity:5.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../models/', help='The path to the saved model')
    parser.add_argument('--dataset_path', type=str, default='../generated_datasets/lsystem_dataset_48267__30_07_2024_12_54', help='The path of the dataset')
    parser.add_argument('--mean', type=float, default=0.9964, help='The mean value of the dataset')
    parser.add_argument('--std', type=float, default=0.0602, help='The standard deviation of the dataset')

    # Model parameters (same as train.py)
    parser.add_argument('--embed_size', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden state dimension')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

    parsed_args = parser.parse_args()
    print(parsed_args)
    test(parsed_args)
