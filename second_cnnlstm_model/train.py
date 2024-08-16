import argparse
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import second_cnnlstm_model.utils as utils
from second_cnnlstm_model.lsystem_dataloaders import get_train_loader, get_valid_loader
from second_cnnlstm_model.model import EncoderCNN, DecoderRNN
from second_cnnlstm_model.vocabulary import Vocabulary
from second_cnnlstm_model.metrics import AverageMetric
import second_cnnlstm_model.metrics as metrics


def train(args):
    device = utils.get_device()

    timestamp = time.strftime("%d_%m_%Y_%H_%M")
    model_path = os.path.join(args.model_path, f"run_{timestamp}")
    runs_path = os.path.join(args.tb_path, f"run_{timestamp}")

    if args.load_path == '':
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        os.mkdir(model_path)

        if not os.path.exists(args.tb_path):
            os.makedirs(args.tb_path)
        os.mkdir(runs_path)

    transform = transforms.Compose([
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

        current_run_dir = os.path.basename(os.path.dirname(args.load_path))
        model_path = str(os.path.join(args.model_path, current_run_dir))
        runs_path = str(os.path.join(args.tb_path, current_run_dir))

    writer = SummaryWriter(runs_path)
    encoder.train()
    decoder.train()

    train_loss = AverageMetric()
    train_perplexity = AverageMetric()
    train_bpc = AverageMetric()
    train_percentage_correct = AverageMetric()
    train_percentage_false_syntax = AverageMetric()
    train_percentage_non_terminated = AverageMetric()
    train_percentage_residue = AverageMetric()
    train_hausdorff_distance = AverageMetric()

    valid_loss = AverageMetric()
    valid_perplexity = AverageMetric()
    valid_bpc = AverageMetric()
    valid_percentage_correct = AverageMetric()
    valid_percentage_false_syntax = AverageMetric()
    valid_percentage_non_terminated = AverageMetric()
    valid_percentage_residue = AverageMetric()
    valid_hausdorff_distance = AverageMetric()

    total_batches = len(train_dataloader)
    for epoch in range(starting_epoch, args.num_epochs):
        train_loss.reset()
        train_perplexity.reset()
        train_bpc.reset()
        train_percentage_correct.reset()
        train_percentage_false_syntax.reset()
        train_percentage_non_terminated.reset()
        train_percentage_residue.reset()
        train_hausdorff_distance.reset()

        if args.dataset_version == 2:
            train_dataloader.dataset.set_epoch(epoch)

        # Training
        for i, (images, captions, lengths, angles, distances) in enumerate(train_dataloader):
            images = images.to(device)
            captions = captions.to(device)
            lengths = [x - 1 for x in lengths]
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions[:, :-1], lengths)

            loss = loss_fn(outputs, targets)
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.add_value(loss.item())
            train_perplexity.add_value(np.exp(loss.item()))
            train_bpc.add_value(loss.item()/np.log(2))

            converted_targets = metrics.convert_packed_padded_sequence(targets, lengths, vocabulary=vocab)
            converted_outputs = metrics.convert_packed_padded_sequence(outputs, lengths, vocabulary=vocab, convert_predictions=True)
            percentage_correct, percentage_false_syntax, percentage_non_terminated, percentage_residue = metrics.compute_correctness_metrics(converted_outputs, converted_targets, angles, distances, strict=True)
            mean_hausdorff_distance = metrics.compute_hausdorff_metric(converted_outputs, converted_targets, angles, distances, normalize=False)
            train_percentage_correct.add_value(percentage_correct)
            train_percentage_false_syntax.add_value(percentage_false_syntax)
            train_percentage_non_terminated.add_value(percentage_non_terminated)
            train_percentage_residue.add_value(percentage_residue)
            train_hausdorff_distance.add_value(mean_hausdorff_distance)

            if (i + 1) % args.log_step == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{total_batches}],"
                      f" Average Loss: {train_loss.average_value:.4f}, Average Perplexity: {train_perplexity.average_value:5.4f},"
                      f" Average BPC: {train_bpc.average_value:5.4f},"
                      f" Average % correct: {train_percentage_correct.average_value:2.2f}, Average % false syntax: {train_percentage_false_syntax.average_value:2.2f},"
                      f" Average % non-terminated: {train_percentage_non_terminated.average_value:2.2f}, Average % residue: {train_percentage_residue.average_value:2.2f},"
                      f" Average Hausdorff distance: {train_hausdorff_distance.average_value:5.4f}")
                writer.add_scalar("Average Training Loss", train_loss.average_value, global_step=epoch * total_batches + i + 1)
                writer.add_scalar("Average Training Perplexity", train_perplexity.average_value, global_step=epoch * total_batches + i + 1)
                writer.add_scalar("Average Training BPC", train_bpc.average_value, global_step=epoch * total_batches + i + 1)
                writer.add_scalar("Average Training % correct", train_percentage_correct.average_value, global_step=epoch * total_batches + i + 1)
                writer.add_scalar("Average Training % false syntax", train_percentage_false_syntax.average_value, global_step=epoch * total_batches + i + 1)
                writer.add_scalar("Average Training % non-terminated", train_percentage_non_terminated.average_value, global_step=epoch * total_batches + i + 1)
                writer.add_scalar("Average Training % residue", train_percentage_residue.average_value, global_step=epoch * total_batches + i + 1)
                writer.add_scalar("Average Training Hausdorff distance", train_hausdorff_distance.average_value, global_step=epoch * total_batches + i + 1)
                writer.flush()

                train_loss.reset()
                train_perplexity.reset()
                train_bpc.reset()
                train_percentage_correct.reset()
                train_percentage_false_syntax.reset()
                train_percentage_non_terminated.reset()
                train_percentage_residue.reset()
                train_hausdorff_distance.reset()

        utils.save_checkpoint(os.path.join(model_path, f"model-{epoch + 1}.pth.tar"), encoder, decoder, optimizer, epoch)

        # Validation
        encoder.eval()
        decoder.eval()

        if args.dataset_version == 2:
            valid_dataloader.dataset.set_epoch(epoch)

        valid_loss.reset()
        valid_perplexity.reset()
        valid_bpc.reset()
        valid_percentage_correct.reset()
        valid_percentage_false_syntax.reset()
        valid_percentage_non_terminated.reset()
        valid_percentage_residue.reset()
        valid_hausdorff_distance.reset()

        with torch.no_grad():
            for i, (images, captions, lengths, angles, distances) in enumerate(valid_dataloader):
                images = images.to(device)
                captions = captions.to(device)
                max_sequence_length = captions.size()[-1]

                features = encoder(images)
                outputs = decoder.generate_caption(features, max_sequence_length-1, bos_token=torch.tensor([vocab('<bos>')]).to(device), return_idx=False)

                loss = valid_loss_fn(outputs.view(-1, outputs.size(dim=-1)), captions[:, 1:].reshape(-1))
                valid_loss.add_value(loss.item())
                valid_perplexity.add_value(np.exp(loss.item()))
                valid_bpc.add_value(loss.item()/np.log(2))

                converted_targets = metrics.convert_padded_sequence(captions[:, 1:], vocab('<eos>'), vocabulary=vocab)
                converted_outputs = metrics.convert_padded_sequence(outputs, vocab('<eos>'), vocabulary=vocab, convert_predictions=True)
                percentage_correct, percentage_false_syntax, percentage_non_terminated, percentage_residue = metrics.compute_correctness_metrics(converted_outputs, converted_targets, angles, distances, strict=True)
                mean_hausdorff_distance = metrics.compute_hausdorff_metric(converted_outputs, converted_targets, angles, distances, normalize=False)
                valid_percentage_correct.add_value(percentage_correct)
                valid_percentage_false_syntax.add_value(percentage_false_syntax)
                valid_percentage_non_terminated.add_value(percentage_non_terminated)
                valid_percentage_residue.add_value(percentage_residue)
                valid_hausdorff_distance.add_value(mean_hausdorff_distance)

        print(f"Validation for Epoch [{epoch+1}/{args.num_epochs}],"
              f" Average Loss: {valid_loss.average_value:.4f}, Average Perplexity: {valid_perplexity.average_value:5.4f},"
              f" Average BPC: {valid_bpc.average_value:5.4f},"
              f" Average % correct: {valid_percentage_correct.average_value:2.2f}, Average % false syntax: {valid_percentage_false_syntax.average_value:2.2f},"
              f" Average % non-terminated: {valid_percentage_non_terminated.average_value:2.2f}, Average % residue: {valid_percentage_residue.average_value:2.2f},"
              f" Average Hausdorff distance: {valid_hausdorff_distance.average_value:5.4f}")
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
        writer.add_scalars(
            "Average Training BPC vs Average Validation BPC",
            {'Training': train_bpc.previous_value, 'Validation': valid_bpc.average_value},
            global_step=epoch + 1
        )
        writer.add_scalars(
            "Average Training % correct vs Average Validation % correct",
            {'Training': train_percentage_correct.previous_value, 'Validation': valid_percentage_correct.average_value},
            global_step=epoch + 1
        )
        writer.add_scalars(
            "Average Training % false syntax vs Average Validation % false syntax",
            {'Training': train_percentage_false_syntax.previous_value, 'Validation': valid_percentage_false_syntax.average_value},
            global_step=epoch + 1
        )
        writer.add_scalars(
            "Average Training % non-terminated vs Average Validation % non-terminated",
            {'Training': train_percentage_non_terminated.previous_value, 'Validation': valid_percentage_non_terminated.average_value},
            global_step=epoch + 1
        )
        writer.add_scalars(
            "Average Training % residue vs Average Validation % residue",
            {'Training': train_percentage_residue.previous_value, 'Validation': valid_percentage_residue.average_value},
            global_step=epoch + 1
        )
        writer.add_scalars(
            "Average Training Hausdorff distance vs Average Validation Hausdorff distance",
            {'Training': train_hausdorff_distance.previous_value, 'Validation': valid_hausdorff_distance.average_value},
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
    parser.add_argument('--dataset_path', type=str, default='../generated_datasets/lsystem_dataset_v2_48267__01_08_2024_15_55', help='The path of the dataset')
    parser.add_argument('--dataset_version', type=int, default=2, help='The format version of the dataset')
    parser.add_argument('--mean', type=float, default=0.9947, help='The mean value of the dataset')
    parser.add_argument('--std', type=float, default=0.0729, help='The standard deviation of the dataset')
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
