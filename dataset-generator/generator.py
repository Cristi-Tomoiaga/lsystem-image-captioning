import os
import random
import time

import pandas as pd
import numpy as np

from lsystem import LSystem
from lword_renderer import LWordRenderer
from lword_preprocessor import LwordPreprocessor


def split_indices(size: int, split: tuple[float, float, float]) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(size))
    random.shuffle(indices)

    train_split = split[0]
    valid_split = split[1]
    train_end = int(train_split * size)
    valid_end = train_end + int(valid_split * size)

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    return train_indices, valid_indices, test_indices


class DatasetGenerator:
    def __init__(self, lsystem: LSystem, distance: float, angle: tuple[float, float], iterations: tuple[int, int]):
        self.__lsystem = lsystem
        self.__distance = distance
        self.__angle = angle
        self.__iterations = iterations

        self.__renderer = LWordRenderer(512, 512)
        self.__preprocessor = LwordPreprocessor

    def generate(self, size: int, split: tuple[float, float, float], path: str, log_step: int) -> str | None:
        timestamp = time.strftime("%d_%m_%Y_%H_%M")
        timestamp_path = f"{path}_{size}__{timestamp}"

        train_path = os.path.join(timestamp_path, 'train')
        valid_path = os.path.join(timestamp_path, 'valid')
        test_path = os.path.join(timestamp_path, 'test')

        try:
            os.makedirs(train_path)
            os.makedirs(valid_path)
            os.makedirs(test_path)
        except OSError:
            print("Provided path already contains data")
            return

        train_indices, valid_indices, test_indices = split_indices(size, split)

        generated = []
        lwords = []
        images = []
        i = 0
        while i < size:
            num_iterations = random.randint(self.__iterations[0], self.__iterations[1])
            angle = random.uniform(self.__angle[0], self.__angle[1])

            lword = self.__lsystem.generate(num_iterations, clean_lword=True)
            lword = self.__renderer.fix_lword_geometrically(lword, angle, self.__distance)
            lword = self.__preprocessor.process_lword_repeatedly(lword)

            if (lword, angle) in generated:
                continue

            image = self.__renderer.render(lword, angle, self.__distance, rescale=True)
            image_name = f'image_{i}.png'

            if i in train_indices:
                image.save(os.path.join(train_path, image_name))
            elif i in valid_indices:
                image.save(os.path.join(valid_path, image_name))
            elif i in test_indices:
                image.save(os.path.join(test_path, image_name))

            lwords.append(lword)
            images.append(image_name)
            generated.append((lword, angle))

            if (i + 1) % log_step == 0:
                print(f"Generated [{i+1}/{size}] images")

            i += 1

        np_lwords = np.array(lwords)
        np_images = np.array(images)

        train_lwords, train_images = np_lwords[train_indices], np_images[train_indices]
        valid_lwords, valid_images = np_lwords[valid_indices], np_images[valid_indices]
        test_lwords, test_images = np_lwords[test_indices], np_images[test_indices]

        train_data = pd.DataFrame({'lword': train_lwords, 'image': train_images})
        valid_data = pd.DataFrame({'lword': valid_lwords, 'image': valid_images})
        test_data = pd.DataFrame({'lword': test_lwords, 'image': test_images})

        train_data.to_csv(os.path.join(train_path, 'captions.csv'), index=False, header=False)
        valid_data.to_csv(os.path.join(valid_path, 'captions.csv'), index=False, header=False)
        test_data.to_csv(os.path.join(test_path, 'captions.csv'), index=False, header=False)

        return timestamp_path
