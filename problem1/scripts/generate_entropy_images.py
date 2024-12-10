import click
from scripts.utils import get_data_samples_from_path, Sample

from typing import Dict
from collections import Counter, defaultdict
import torch

from torchvision.transforms import ToPILImage
import os
from tqdm import tqdm
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from numba import njit


@njit
def get_numpy_entropy_image(n_blocks: int, probabilities: np.ndarray) -> np.ndarray:
    image = np.zeros((n_blocks, n_blocks))
    entropy = np.zeros(n_blocks)
    kl_divergence = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        value = 0
        p = probabilities[i]
        for k in range(0, 256):
            if p[k] == 0:
                continue
            value += p[k] * np.log2(p[k])
        entropy[i] = value
    for i in range(n_blocks):
        for j in range(n_blocks):
            value = 0
            p = probabilities[i]
            q = probabilities[j]
            for k in range(0, 256):
                if p[k] == 0 or q[k] == 0:
                    continue
                value += q[k] * np.log2(q[k]/p[k])
            kl_divergence[i, j] = value
    for i in range(n_blocks):
        for j in range(n_blocks):
            if i == j:
                image[i, j] = entropy[i]
            else:
                image[i, j] = kl_divergence[i, j]
    return image


def transform_into_entropy_image(sample: Sample) -> torch.Tensor:
    """Transforms a sample into an grayscale entropy image

    Args:
        sample (Sample): a sample object

    Returns:
        torch.Tensor: grayscale entropy image of dimension (1, 256, 256)
    """
    blocks = []
    for i in range(0, len(sample.data), 256):
        block = sample.data[i:i + 256]
        blocks.append(block)
    # we want a total of 256 blocks
    for i in range(len(blocks), 256):
        blocks.append([0]*256)

    n_blocks = len(blocks)
    assert n_blocks == 256, f"Expected 256 blocks, got {n_blocks}"

    probabilities = np.zeros((n_blocks, 256))
    for i, block in enumerate(blocks):
        counts = Counter(block)
        for b, count in counts.items():
            probabilities[i][b] = count / 256
    image = get_numpy_entropy_image(n_blocks, probabilities)
    image = torch.tensor(image, dtype=torch.float32)
    image = torch.clamp(32*image, max=255)
    # add one extra dimension
    image = image.unsqueeze(0)
    return image


def worker(sample, save_path):
    toimage_tf = ToPILImage()
    image_tensor = transform_into_entropy_image(sample)
    image = toimage_tf(image_tensor)
    image.save(f"{save_path}/{sample.name}.png")


@click.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Path to the image dataset')
@click.option('--output', '-o', required=True, type=click.Path(), help='Path to the output directory')
@click.option('--multicore/--no-multicore', default=False, is_flag=True, help='Use multiple cores to generate the images')
def generate_entropy_images(input, output, multicore):
    images_per_class = get_data_samples_from_path(input)
    args_list = []
    for klass, samples in images_per_class.items():
        os.makedirs(f"{output}/{klass}", exist_ok=True)


    print("Generating entropy images")
    if multicore:
        for klass, samples in images_per_class.items():
            args_list.extend([(sample, f"{output}/{klass}")
                            for sample in samples])
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(worker, *args)
                    for args in args_list]
            results = [future.result() for future in futures]
    else:
        for klass, samples in images_per_class.items():
            for sample in tqdm(samples, desc=f"Processing samples from {klass}"):
                worker(sample, f"{output}/{klass}")

def main():
    generate_entropy_images()

if __name__ == '__main__':
    main()