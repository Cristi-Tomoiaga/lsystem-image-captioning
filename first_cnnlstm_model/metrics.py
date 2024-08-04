import math

import numpy as np
from skimage import metrics as skimage_metrics

from dataset_generator.lword_preprocessor import LWordPreprocessor
from dataset_generator.lword_renderer import LWordRenderer


class AverageMetric:
    def __init__(self):
        self.__total = 0
        self.__running_sum = 0.0
        self.__average_value = 0.0
        self.__previous_average_value = 0.0

    def reset(self):
        self.__previous_average_value = self.__average_value
        self.__total = 0
        self.__running_sum = 0.0
        self.__average_value = 0.0

    @property
    def average_value(self):
        return self.__average_value

    @property
    def previous_value(self):
        return self.__previous_average_value

    def add_value(self, value, count=1):
        self.__total += count
        self.__running_sum += value * count
        self.__average_value = self.__running_sum / self.__total


def check_lword_syntax(lword, angle, distance, strict):
    lword = lword.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '')
    renderer = LWordRenderer(512, 512)

    if not LWordPreprocessor.check_syntax(lword):
        return False

    if strict:
        if renderer.validate_lword_geometrically(lword, angle, distance):
            return False

        if not LWordPreprocessor.check_canceling_rotations(lword):
            return False

        if not LWordPreprocessor.check_empty_branches(lword):
            return False

        if not LWordPreprocessor.check_ordered_branches(lword):
            return False

        if not LWordPreprocessor.check_ending_subbranches(lword):
            return False

    return True


def compute_hausdorff_distance(output_image, target_image):
    np_output_image = np.array(output_image)
    np_target_image = np.array(target_image)

    binary_output_image = (np_output_image == 0).astype(np.float32)
    binary_target_image = (np_target_image == 0).astype(np.float32)

    return skimage_metrics.hausdorff_distance(binary_output_image, binary_target_image)


def compute_correctness_metrics(outputs, targets, angles, distances, strict=True):
    total = len(targets)
    correct = 0
    false_syntax = 0
    non_terminated = 0
    residue = 0

    for output, target, angle, distance in zip(outputs, targets, angles, distances):
        if output.find('<eos>') == -1:  # improper <bos> is checked in the residue case
            non_terminated += 1
        elif not check_lword_syntax(output, angle.item(), distance.item(), strict):
            false_syntax += 1
        elif output != target:  # the conversion function cuts terminated words at <eos>, correctly placed <pad>s are left out
            residue += 1
        else:
            correct += 1

    percentage_correct = correct / total * 100
    percentage_false_syntax = false_syntax / total * 100
    percentage_non_terminated = non_terminated / total * 100
    percentage_residue = residue / total * 100

    return percentage_correct, percentage_false_syntax, percentage_non_terminated, percentage_residue


def compute_hausdorff_metric(outputs, targets, angles, distances, normalize=False):
    total = 0
    running_sum = 0.0
    renderer = LWordRenderer(512, 512)

    for output, target, angle, distance in zip(outputs, targets, angles, distances):
        output = output.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '')
        target = target.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '')

        if not LWordPreprocessor.check_syntax(output):
            continue

        try:
            output_image = renderer.render(output, angle.item(), distance.item(), rescale=True)
            target_image = renderer.render(target, angle.item(), distance.item(), rescale=True)
        except ZeroDivisionError:
            continue

        np_output_image = np.array(output_image)
        np_target_image = np.array(target_image)

        binary_output_image = (np_output_image == 0).astype(np.float32)
        binary_target_image = (np_target_image == 0).astype(np.float32)

        hausdorff_distance = skimage_metrics.hausdorff_distance(binary_output_image, binary_target_image)
        running_sum += hausdorff_distance

        total += 1

    if total == 0:
        mean_hausdorff_distance = math.sqrt(2 * 512 * 512)  # Investigate ignoring the batch
    else:
        mean_hausdorff_distance = running_sum / total

    if normalize:
        mean_hausdorff_distance = mean_hausdorff_distance / math.sqrt(2 * 512 * 512)

    return mean_hausdorff_distance


def convert_packed_padded_sequence(packed_padded, lengths, vocabulary=None, convert_predictions=False):
    packed_padded = packed_padded.detach().cpu()

    if convert_predictions:
        packed_padded = packed_padded.argmax(dim=-1)

    num_sequences = len(lengths)
    sequences = [[] for _ in range(num_sequences)]

    i = 0
    for step in range(max(lengths)):
        for j in range(num_sequences):
            if step < lengths[j]:
                sequences[j].append(packed_padded[i].item())

                i += 1

    if vocabulary is not None:
        sequences = list(map(vocabulary.convert_to_lword, sequences))

    return sequences


def convert_padded_sequence(padded, end_token, vocabulary=None, convert_predictions=False):
    padded = padded.detach().cpu()

    if convert_predictions:
        padded = padded.argmax(dim=-1)

    sequences = []
    for seq in padded:
        end_token_index = (seq == end_token).nonzero(as_tuple=True)[0]

        if end_token_index.numel() > 0:
            seq = seq[:end_token_index[0].item()+1]

        sequences.append(seq.tolist())

    if vocabulary is not None:
        sequences = list(map(vocabulary.convert_to_lword, sequences))

    return sequences
