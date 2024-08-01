import argparse
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import first_cnnlstm_model.utils as utils
from first_cnnlstm_model.lsystem_dataloaders import get_train_loader, get_valid_loader
from first_cnnlstm_model.model import EncoderCNN, DecoderRNN
from first_cnnlstm_model.vocabulary import Vocabulary
from first_cnnlstm_model.metrics import AverageMetric


def train(args):
    device = utils.get_device()

    timestamp = time.strftime("%d_%m_%Y_%H_%M")
    model_path = os.path.join(args.model_path, f"run_{timestamp}")
    runs_path = os.path.join(args.tb_path, f"run_{timestamp}")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    os.mkdir(model_path)

    if not os.path.exists(args.tb_path):
        os.makedirs(args.tb_path)
    os.mkdir(runs_path)

    transform = transforms.Compose([  # Investigate RandomCrop, RandomHorizontalFlip, angle changing?
        transforms.ToTensor(),
        transforms.Normalize((args.mean,), (args.std,))
    ])

    vocab = Vocabulary()

    train_dataloader = get_train_loader(
        root_dir=args.dataset_path,
        version=args.dataset_version,
        transform=transform,
        vocabulary=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    valid_dataloader = get_valid_loader(
        root_dir=args.dataset_path,
        version=args.dataset_version,
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
        vocab_size=len(vocab),
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    loss_fn = nn.CrossEntropyLoss()
    valid_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab("<pad>"))
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    starting_epoch = 0
    if args.load_path != '':
        checkpoint = utils.load_checkpoint(args.load_path, move_to_cpu=False)

        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        starting_epoch = checkpoint["epoch"] + 1

    writer = SummaryWriter(runs_path)
    encoder.train()
    decoder.train()

    train_loss = AverageMetric()
    train_perplexity = AverageMetric()

    valid_loss = AverageMetric()
    valid_perplexity = AverageMetric()

    total_batches = len(train_dataloader)
    for epoch in range(starting_epoch, args.num_epochs):
        train_loss.reset()
        train_perplexity.reset()

        if args.dataset_version == 2:
            train_dataloader.dataset.set_epoch(epoch)

        # Training
        for i, (images, captions, lengths, _, _) in enumerate(train_dataloader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions, lengths)

            loss = loss_fn(outputs, targets)
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.add_value(loss.item())
            train_perplexity.add_value(np.exp(loss.item()))
            if (i + 1) % args.log_step == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{total_batches}],"
                      f" Average Loss: {train_loss.average_value:.4f}, Average Perplexity: {train_perplexity.average_value:5.4f}")
                writer.add_scalar("Average Training Loss", train_loss.average_value, global_step=epoch * total_batches + i + 1)
                writer.add_scalar("Average Training Perplexity", train_perplexity.average_value, global_step=epoch * total_batches + i + 1)
                writer.flush()

                train_loss.reset()
                train_perplexity.reset()

        utils.save_checkpoint(os.path.join(model_path, f"model-{epoch + 1}.pth.tar"), encoder, decoder, optimizer, epoch)

        # Validation
        encoder.eval()
        decoder.eval()

        if args.dataset_version == 2:
            valid_dataloader.dataset.set_epoch(epoch)

        valid_loss.reset()  # investigate balancing loss with batch size
        valid_perplexity.reset()
        with torch.no_grad():
            for i, (images, captions, lengths, _, _) in enumerate(valid_dataloader):
                images = images.to(device)
                captions = captions.to(device)
                max_sequence_length = captions.size()[-1]

                features = encoder(images)
                outputs = decoder.generate_caption(features, max_sequence_length, return_idx=False)

                loss = valid_loss_fn(outputs.view(-1, outputs.size(dim=-1)), captions.view(-1))
                valid_loss.add_value(loss.item())
                valid_perplexity.add_value(np.exp(loss.item()))

        print(f"Validation for Epoch [{epoch+1}/{args.num_epochs}],"
              f" Average Loss: {valid_loss.average_value:.4f}, Average Perplexity: {valid_perplexity.average_value:5.4f}")
        writer.add_scalars(
            "Average Training Loss vs Average Validation Loss",
            {'Training': train_loss.previous_value, 'Validation': valid_loss.average_value},
            global_step=epoch + 1
        )
        writer.add_scalars(
            "Average Training Perplexity vs Average Validation Perplexity",
            {'Training': train_perplexity.previous_value, 'Validation': valid_perplexity.average_value},
            global_step=epoch + 1
        )
        writer.flush()

        encoder.train()
        decoder.train()

    print('Finished training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../models/', help='The path for saving trained models')
    parser.add_argument("--load_path", type=str, default='', help='The path for loading and resuming training')
    parser.add_argument("--tb_path", type=str, default="../runs/", help='The path for saving tensorboard logs')
    parser.add_argument('--dataset_path', type=str, default='../generated_datasets/lsystem_dataset_v2_20__31_07_2024_20_01', help='The path of the dataset')
    parser.add_argument('--dataset_version', type=int, default=2, help='The format version of the dataset')
    parser.add_argument('--mean', type=float, default=0.9964, help='The mean value of the dataset')
    parser.add_argument('--std', type=float, default=0.0602, help='The standard deviation of the dataset')
    parser.add_argument('--log_step', type=int, default=10, help='The step size for printing log info')  # 10

    # Model parameters                                                                      Tutorial, Alternative, Paper
    parser.add_argument('--embed_size', type=int, default=128, help='Embedding dimension')  # 256, 256, 256
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden state dimension')  # 512, 256, 256

    parser.add_argument('--num_epochs', type=int, default=495, help='Number of epochs')  # 5, 100, 495
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')  # 128, 32
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')  # 2, 8
    parser.add_argument('--learning_rate', type=float, default=0.00025, help='Learning rate')  # 0.001, 3e-4, 0.00025

    parsed_args = parser.parse_args()
    print(parsed_args)
    train(parsed_args)
