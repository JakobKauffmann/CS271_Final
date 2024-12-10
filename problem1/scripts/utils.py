import os
from dataclasses import dataclass
from typing import Dict, List
from tqdm import tqdm


@dataclass
class Sample:
    id: int  # something like 0, 1, 2
    klass: str  # the klass type, like A, B, C, D, E, unknown
    data: List  # the list of integers
    name: str  # the name of the original sample. For example U_0, A_1, etc.


def get_data_samples_from_path(path: str) -> Dict[str, List[Sample]]:
    """Returns all the samples from the given path, organized by class

    Args:
        path (str): the path to the ENTIRE dataset

    Returns:
        Dict[str, List[Sample]]: a dictionary where the key is the class name
        and the value is a list of samples of that class
    """
    # get the name of the immediate subdirectories
    subdir_paths = [f.path for f in os.scandir(path) if f.is_dir()]
    sample_prefix_to_class = {
        "A": "A",
        "U": "unknown",
        "B": "B",
        "C": "C",
        "D": "D",
        "E": "E",
    }
    result_objects = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "E": [],
        "unknown": [],
    }
    for subdir_path in subdir_paths:
        print(f"Processing path: {subdir_path}")
        subdir_name = os.path.basename(subdir_path)
        assert subdir_name in result_objects, f"Unknown class {subdir_name}"
        samples = [f.path for f in os.scandir(subdir_path) if f.is_file()]
        filtered_samples = []
        for sample in samples:
            name = os.path.basename(sample)
            prefix, sample_id = name.split("_")
            if prefix not in sample_prefix_to_class:
                print(
                    f"Skipping {sample} because it is not a valid prefix: {prefix}")
                continue
            filtered_samples.append(sample)
        for sample in tqdm(filtered_samples):
            name = os.path.basename(sample)
            prefix, sample_id = name.split("_")
            # we need to read this sample. The sample is simply
            # a file where each line is a integer
            with open(sample, "r") as f:
                lines = f.readlines()
                lines = [int(x) for x in lines]
            # assert the number of expected integers
            assert len(
                lines) == 51_200, f"Expected 51_200 integers, got {len(lines)}"
            result_objects[subdir_name].append(
                Sample(int(sample_id), subdir_name, lines, name=name))
    return result_objects
