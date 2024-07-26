import argparse

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

import utils
from model import EncoderCNN, DecoderRNN
from vocabulary import Vocabulary


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize((512, 512), Image.Resampling.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def generate(image_path, model_path, embed_size, hidden_size, mean, std, max_sequence_length):
    device = utils.get_device()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    vocab = Vocabulary()

    encoder = EncoderCNN(feature_size=embed_size)
    decoder = DecoderRNN(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        max_sequence_length=max_sequence_length
    )

    checkpoint = utils.load_checkpoint(model_path, move_to_cpu=True)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    image = load_image(image_path, transform)
    image = image.to(device)

    with torch.no_grad():
        features = encoder(image)
        generated_idx = decoder.generate_caption(features)  # (1, max_sequence_length)
        lword = vocab.convert_to_lword(generated_idx[0].cpu().numpy())  # (max_sequence_length)

    return lword


def main(args):
    lword = generate(args.image_path, args.model_path, args.embed_size, args.hidden_size, args.mean, args.std, args.max_sequence_length)
    image = Image.open(args.image_path)

    print(f"Generated lword: {lword}")
    plt.imshow(np.asarray(image), cmap="gray")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Input image path')
    parser.add_argument('--model_path', type=str, default='../models/run_26_07_2024_13_33/model-5.pth.tar', help='The path to the saved model')
    parser.add_argument('--mean', type=float, default=0.0, help='The mean value of the dataset')
    parser.add_argument('--std', type=float, default=1.0, help='The standard deviation of the dataset')
    parser.add_argument('--max_sequence_length', type=int, default=60, help='The maximum sequence length of the dataset')

    # Model parameters (same as train.py)
    parser.add_argument('--embed_size', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden state dimension')

    parsed_args = parser.parse_args()
    print(parsed_args)
    main(parsed_args)
