from numba import njit
from scripts.utils import get_data_samples_from_path, Sample
import numpy as np

import click
import os
from torchvision.transforms import ToPILImage
import torch
from tqdm import tqdm

@njit
def digraph_frequency(sequence: np.ndarray) -> np.ndarray:
    """Computes the frequency of each pair of elements in the sequence

    Args:
        sequence (np.ndarray): the sequence of elements

    Returns:
        np.ndarray: a 256x256 matrix where the element (i, j) is the frequency
        of the pair (i, j) in the sequence
    """
    # find the number of times a value i is followed by a value j
    # in the sequence
    frequency = np.zeros((256, 256), dtype=np.int32)
    for _i in range(len(sequence) - 1):
        _j = _i + 1
        i = sequence[_i]
        j = sequence[_j]
        frequency[i, j] += 1
    image = np.zeros((256, 256))
    row_sum = frequency.sum(axis=1)
    # normalize the frequency matrix only in locations where the row sum is not zero
    for i in range(256):
        if row_sum[i] == 0:
            continue
        image[i] = frequency[i] / row_sum[i]

    # scale the image to 0-255
    return image*255


def worker(sample, save_path):
    toimage_tf = ToPILImage()
    image_np = digraph_frequency(np.array(sample.data))
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)
    image = toimage_tf(image_tensor)
    image.save(f"{save_path}/{sample.name}.png")


@click.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Path to the image dataset')
@click.option('--output', '-o', required=True, type=click.Path(), help='Path to the output directory')
def generate_digraph_images(input, output):
    images_per_class = get_data_samples_from_path(input)
    for klass, samples in images_per_class.items():
        os.makedirs(f"{output}/{klass}", exist_ok=True)

    print("Generating digraph images")
    for klass, samples in images_per_class.items():
        for sample in tqdm(samples, desc=f"Processing samples from {klass}"):
            worker(sample, f"{output}/{klass}")


def main():
    generate_digraph_images()


if __name__ == '__main__':
    main()
