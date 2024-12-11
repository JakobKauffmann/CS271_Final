import torchvision
import os


class ExcludeUnknownDataset(torchvision.datasets.DatasetFolder):
    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(
            directory) if entry.is_dir())
        classes = [c for c in classes if c != "unknown"]
        if not classes:
            raise FileNotFoundError(
                f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
