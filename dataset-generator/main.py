import argparse

from generator import DatasetGenerator
from lsystem import LSystem


def main(args):
    lsystem = LSystem(
        "FX",
        [
            "X:0.34->[-FX][+FX]",
            "X:0.33->-FX",
            "X:0.33->+FX",
        ],
        True
    )

    dataset_generator = DatasetGenerator(
        lsystem,
        args.distance,
        (args.angle_min, args.angle_max),
        (args.it_min, args.it_max)
    )
    path = dataset_generator.generate(args.size, (args.train_size, args.valid_size, args.test_size), args.path, args.log_step)

    if path is not None:
        print(f"Generated and saved the dataset to {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../generated_datasets/lsystem_dataset', help='The path where the dataset is saved')
    parser.add_argument('--log_step', type=int, default=5000, help='The step size for printing log info')

    parser.add_argument('--size', type=int, default=20, help='The size of the dataset')
    parser.add_argument('--train_size', type=float, default=0.9, help='Split percentage for training')
    parser.add_argument('--valid_size', type=float, default=0.05, help='Split percentage for validation')
    parser.add_argument('--test_size', type=float, default=0.05, help='Split percentage for testing')

    parser.add_argument('--distance', type=float, default=100, help='Distance used for drawing F')
    parser.add_argument('--angle_min', type=float, default=15, help='Minimum turning angle')
    parser.add_argument('--angle_max', type=float, default=40, help='Maximum turning angle')  # 60
    parser.add_argument('--it_min', type=int, default=1, help='Minimum number of iterations')
    parser.add_argument('--it_max', type=int, default=7, help='Maximum number of iterations')

    parsed_args = parser.parse_args()
    print(parsed_args)
    main(parsed_args)
