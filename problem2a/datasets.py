import torchvision
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from tqdm import tqdm


def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(
        directory) if entry.is_dir())
    classes = [c for c in classes if c != "unknown"]
    if not classes:
        raise FileNotFoundError(
            f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class ExcludeUnknownDataset(torchvision.datasets.DatasetFolder):
    def find_classes(self, directory):
        classes, class_to_idx = find_classes(directory)
        return classes, class_to_idx

class InMemoryExcludeUnknownDataset(Dataset):

    def __init__(self, root, transform):
        self.images = []
        self.labels = []
        self.transform = transform
        classes, class_to_idx = find_classes(root)
        print(f"Loading {len(classes)} classes onto memory")
        for target_class in classes:
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(root, target_class)
            print(f"Loading class {target_class!r} from {target_dir}")
            for inner_root, _, fnames in sorted(os.walk(target_dir)):
                for fname in tqdm(sorted(fnames)):
                    path = os.path.join(inner_root, fname)
                    image = Image.open(path)
                    image = transform(image)
                    self.images.append(image)
                    self.labels.append(class_index)
        # stack all images into a single tensor
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]