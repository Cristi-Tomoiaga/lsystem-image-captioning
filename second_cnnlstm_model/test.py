import argparse

import numpy as np
import torch
from torchvision import transforms
from torch import nn

import second_cnnlstm_model.utils as utils
from second_cnnlstm_model.vocabulary import Vocabulary
from second_cnnlstm_model.model import EncoderCNN, DecoderRNN
from second_cnnlstm_model.lsystem_dataloaders import get_test_loader
from second_cnnlstm_model.metrics import AverageMetric
import second_cnnlstm_model.metrics as metrics


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
    test_bpc = AverageMetric()
    test_percentage_correct = AverageMetric()
    test_percentage_false_syntax = AverageMetric()
    test_percentage_non_terminated = AverageMetric()
    test_percentage_residue = AverageMetric()
    test_hausdorff_distance = AverageMetric()

    log_loss = AverageMetric()
    log_perplexity = AverageMetric()
    log_bpc = AverageMetric()
    log_percentage_correct = AverageMetric()
    log_percentage_false_syntax = AverageMetric()
    log_percentage_non_terminated = AverageMetric()
    log_percentage_residue = AverageMetric()
    log_hausdorff_distance = AverageMetric()

    test_loss.reset()
    test_perplexity.reset()
    test_bpc.reset()
    test_percentage_correct.reset()
    test_percentage_false_syntax.reset()
    test_percentage_non_terminated.reset()
    test_percentage_residue.reset()
    test_hausdorff_distance.reset()

    total_batches = len(test_dataloader)
    with torch.no_grad():
        for epoch in range(num_epochs):
            if dataset_version == 2:
                test_dataloader.dataset.set_epoch(epoch)

            log_loss.reset()
            log_perplexity.reset()
            log_bpc.reset()
            log_percentage_correct.reset()
            log_percentage_false_syntax.reset()
            log_percentage_non_terminated.reset()
            log_percentage_residue.reset()
            log_hausdorff_distance.reset()

            for i, (images, captions, lengths, angles, distances) in enumerate(test_dataloader):
                images = images.to(device)
                captions = captions.to(device)
                max_sequence_length = captions.size()[-1]

                features = encoder(images)
                outputs = decoder.generate_caption(features, max_sequence_length-1, bos_token=vocab('<bos>'), return_idx=False)

                loss = test_loss_fn(outputs.view(-1, outputs.size(dim=-1)), captions[:, 1:].reshape(-1))
                test_loss.add_value(loss.item())
                test_perplexity.add_value(np.exp(loss.item()))
                test_bpc.add_value(loss.item()/np.log(2))
                log_loss.add_value(loss.item())
                log_perplexity.add_value(np.exp(loss.item()))
                log_bpc.add_value(loss.item()/np.log(2))

                converted_targets = metrics.convert_padded_sequence(captions[:, 1:], vocab('<eos>'), vocabulary=vocab)
                converted_outputs = metrics.convert_padded_sequence(outputs, vocab('<eos>'), vocabulary=vocab, convert_predictions=True)
                percentage_correct, percentage_false_syntax, percentage_non_terminated, percentage_residue = metrics.compute_correctness_metrics(converted_outputs, converted_targets, angles, distances, strict=True)
                mean_hausdorff_distance = metrics.compute_hausdorff_metric(converted_outputs, converted_targets, angles, distances, normalize=False)
                test_percentage_correct.add_value(percentage_correct)
                test_percentage_false_syntax.add_value(percentage_false_syntax)
                test_percentage_non_terminated.add_value(percentage_non_terminated)
                test_percentage_residue.add_value(percentage_residue)
                test_hausdorff_distance.add_value(mean_hausdorff_distance)
                log_percentage_correct.add_value(percentage_correct)
                log_percentage_false_syntax.add_value(percentage_false_syntax)
                log_percentage_non_terminated.add_value(percentage_non_terminated)
                log_percentage_residue.add_value(percentage_residue)
                log_hausdorff_distance.add_value(mean_hausdorff_distance)

                if (i + 1) % args.log_step == 0:
                    print(f"Test - Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_batches}],"
                          f" Average Loss: {log_loss.average_value:.4f}, Average Perplexity: {log_perplexity.average_value:5.4f},"
                          f" Average BPC: {log_bpc.average_value:5.4f},"
                          f" Average % correct: {log_percentage_correct.average_value:2.2f}, Average % false syntax: {log_percentage_false_syntax.average_value:2.2f},"
                          f" Average % non-terminated: {log_percentage_non_terminated.average_value:2.2f}, Average % residue: {log_percentage_residue.average_value:2.2f},"
                          f" Average Hausdorff distance: {log_hausdorff_distance.average_value:5.4f}")

                    log_loss.reset()
                    log_perplexity.reset()
                    log_bpc.reset()
                    log_percentage_correct.reset()
                    log_percentage_false_syntax.reset()
                    log_percentage_non_terminated.reset()
                    log_percentage_residue.reset()
                    log_hausdorff_distance.reset()

            print(f"Test - Finished Epoch [{epoch + 1}/{num_epochs}]")

    print(f"Test results:"
          f" Average Loss: {test_loss.average_value:.4f}, Average Perplexity: {test_perplexity.average_value:5.4f},"
          f" Average BPC: {test_bpc.average_value:5.4f},"
          f" Average % correct: {test_percentage_correct.average_value:2.2f}, Average % false syntax: {test_percentage_false_syntax.average_value:2.2f},"
          f" Average % non-terminated: {test_percentage_non_terminated.average_value:2.2f}, Average % residue: {test_percentage_residue.average_value:2.2f},"
          f" Average Hausdorff distance: {test_hausdorff_distance.average_value:5.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../models/', help='The path to the saved model')
    parser.add_argument('--dataset_path', type=str, default='../generated_datasets/lsystem_dataset_v2_48267__01_08_2024_15_55', help='The path of the dataset')
    parser.add_argument('--epochs', type=int, default=495, help='The number of augmentation epochs')
    parser.add_argument('--mean', type=float, default=0.9947, help='The mean value of the dataset')
    parser.add_argument('--std', type=float, default=0.0729, help='The standard deviation of the dataset')
    parser.add_argument('--log_step', type=int, default=10, help='The step size for printing log info')  # 10

    # Model parameters (same as train.py)
    parser.add_argument('--embed_size', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden state dimension')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

    parsed_args = parser.parse_args()
    print(parsed_args)
    test(parsed_args)
