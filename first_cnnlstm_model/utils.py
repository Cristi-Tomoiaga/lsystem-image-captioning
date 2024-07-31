import torch.optim
# import torch_directml
from prettytable import PrettyTable
from torch import nn
from torch.utils.data import ConcatDataset
from torchvision import transforms

from first_cnnlstm_model.lsystem_dataset import LSystemDataset
from first_cnnlstm_model.vocabulary import Vocabulary


def get_device():
    # =========================== IMPORTANT: change this when changing computers =======================================
    # device = torch_directml.device() if torch_directml.is_available() else torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ==================================================================================================================

    return device


def count_parameters(model: nn.Module):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        params = parameter.numel()
        table.add_row([name, params])
        total_params += params

    print(table)
    print(f"Total Trainable Params: {total_params}")

    return total_params


def save_checkpoint(model_path: str, encoder: nn.Module, decoder: nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    checkpoint = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }

    torch.save(checkpoint, model_path)
    print(f'Saved checkpoint')


def load_checkpoint(model_path: str, move_to_cpu: bool):
    if move_to_cpu:
        return torch.load(model_path, map_location='cpu')
    else:
        return torch.load(model_path)


def compute_mean_std_for_dataset(dataset_type, root_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = LSystemDataset(dataset_type=dataset_type, root_dir=root_dir,
                             vocabulary=Vocabulary(), transform=transform)

    num_pixels = len(dataset) * 512 * 512

    total_sum = 0.0
    for image, _ in dataset:
        total_sum += image[0].sum()
    mean = total_sum / num_pixels

    total_sq_sum = 0.0
    for image, _ in dataset:
        total_sq_sum += ((image[0] - mean).pow(2)).sum()
    # noinspection PyTypeChecker
    std = torch.sqrt(total_sq_sum / (num_pixels - 1))

    return mean, std


def compute_max_sequence_length_for_dataset(dataset_type, root_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = LSystemDataset(dataset_type=dataset_type, root_dir=root_dir,
                             vocabulary=Vocabulary(), transform=transform)

    target_lengths = [len(target) for _, target in ConcatDataset([dataset])]

    return max(target_lengths)
