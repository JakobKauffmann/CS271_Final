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

    probabilities_by_blocks = {}
    for i, block in enumerate(blocks):
        counts = Counter(block)
        probs = defaultdict(float)
        for b, count in counts.items():
            probs[b] = count / 256
        probabilities_by_blocks[i] = probs
    image = torch.zeros((256, 256))
    entropy = torch.zeros(n_blocks)
    for i in range(n_blocks):
        value = 0
        p = probabilities_by_blocks[i]
        for k in range(0, 256):
            if p[k] == 0:
                continue
            value += p[k] * np.log2(p[k])
        entropy[i] = value
    kl_divergence = torch.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            value = 0
            p = probabilities_by_blocks[i]
            q = probabilities_by_blocks[j]
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
    image = torch.clamp(32*image, max=255)
    # add one extra dimension
    image = image.unsqueeze(0)
    return image


def worker(samples, images_dir):
    toimage_tf = ToPILImage()
    os.makedirs(images_dir, exist_ok=True)
    for sample in tqdm(samples):
        image_tensor = transform_into_entropy_image(sample)
        image = toimage_tf(image_tensor)
        image.save(f"{images_dir}/{sample.name}.png")

@click.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Path to the image dataset')
@click.option('--output', '-o', required=True, type=click.Path(), help='Path to the output directory')
def generate_entropy_images(input, output):
    images_per_class = get_data_samples_from_path(input)
    args_list = [
        (samples, f"{output}/{klass}")
        for klass, samples in images_per_class.items()
    ]
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(worker, *args)
                   for args in args_list]
        results = [future.result() for future in futures]

def main():
    generate_entropy_images()

if __name__ == '__main__':
    main()