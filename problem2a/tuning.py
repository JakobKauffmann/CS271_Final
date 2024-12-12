import ray
from torchvision import transforms
from problem2a.datasets import ExcludeUnknownDataset, InMemoryExcludeUnknownDataset
from problem2a.models import CNN2D
from torch.utils.data import random_split
import torch
from contextlib import contextmanager
import os

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import CheckpointConfig
import click
from PIL import Image

import json

def load_data(data_dir: str, resize: bool = False, in_memory: bool = False):
    if resize:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((100, 100))
            ]
        )
        h, w = 100, 100
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
            ]
        )
        h, w = 256, 256

    if in_memory:
        dataset = InMemoryExcludeUnknownDataset(
            root=data_dir,
            transform=transform
        )
    else:
        dataset = ExcludeUnknownDataset(
            root=data_dir,
            transform=transform,
            loader=Image.open,
            extensions=(".png",)
            )
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    manual_seed_gen = torch.Generator().manual_seed(420)
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=manual_seed_gen
    )
    return train_dataset, test_dataset, h, w
    


class TuneCNN(ray.tune.Trainable):

    def setup(self, config):
        self.data_dir = config["data_dir"]
        self.resize = config.get("resize", False)
        self.trainset, self.testset, h, w = load_data(
            self.data_dir,
            resize=self.resize,
            in_memory=config.get("in_memory", False)
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CNN2D(
            num_conv_blocks=config["num_conv_blocks"],
            fc_hidden_units=config["fc_hidden_units"],
            image_height=h,
            image_width=w,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config["lr"]
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=config["batch_size"], shuffle=True
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=False
        )
        self.model.to(self.device)


    @contextmanager
    def eval_context(self):
        try:
            with torch.no_grad():
                self.model.eval()
                yield
        finally:
            self.model.train()


    def calculate_loss_and_accuracy(self, loader):
        correct = 0
        total = 0
        cummulative_loss = 0.0
        n_batches = 0
        with self.eval_context():
            for image_batch, label_batch in loader:
                image_batch, label_batch = image_batch.to(self.device), label_batch.to(self.device)
                outputs = self.model(image_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += label_batch.size(0)
                loss = self.criterion(outputs, label_batch)
                cummulative_loss += loss.item()
                correct += (predicted == label_batch).sum().item()
                n_batches += 1
        average_batch_loss = cummulative_loss / n_batches
        accuracy = correct / total
        return average_batch_loss, accuracy
    
    def step(self):
        # do one epoch on the training loader
        self.model.train()
        for image_batch, label_batch in self.trainloader:
            image_batch, label_batch = image_batch.to(self.device), label_batch.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(image_batch)
            loss = self.criterion(output, label_batch)
            loss.backward()
            self.optimizer.step()

        training_loss, training_accuracy = self.calculate_loss_and_accuracy(self.trainloader)
        testing_loss, testing_accuracy = self.calculate_loss_and_accuracy(self.testloader)
        avg_loss = (training_loss + testing_loss) / 2
        avg_accuracy = (training_accuracy + testing_accuracy) / 2
        return {
            "training_loss": training_loss,
            "training_accuracy": training_accuracy,
            "testing_loss": testing_loss,
            "testing_accuracy": testing_accuracy,
            "avg_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
            }
    
    def save_checkpoint(self, checkpoint_dir):
        checkpoing_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        model_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(checkpoing_data["model_state_dict"], model_path)
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pth")
        torch.save(checkpoing_data["optimizer_state_dict"], optimizer_path)

    def load_checkpoint(self, checkpoint_dir):
        model_path = os.path.join(checkpoint_dir, "model.pth")
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pth")
        self.model.load_state_dict(torch.load(model_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        # make sure the model is back in training mode
        self.model.train()



@click.command()
@click.option(
    "--dataset",
    "-ds",
    help="Path to the dataset",
    required=True
)
@click.option(
    "--save-best-model",
    "-sbm",
    help="Path to save the best model",
    required=True
)
@click.option(
    '--num-samples',
    '-n',
    type=int,
    default=10,
    help='Number of samples to run. This is the number of workers.'
    )
@click.option(
    '--num-epochs',
    '-e',
    type=int,
    default=50,
    help='Number of epochs to run, this is equivalent to the number of steps'
    )
@click.option(
    "--force-cpu",
    is_flag=True,
    help="Force CPU training"
)
@click.option(
    "--trial-name",
    help="Name of the trial"
)
@click.option(
    "--ray-dir",
    help="Ray directory to store results"
)
@click.option(
    "--resize/--no-resize",
    is_flag=True,
    help="Resize the images to 100x100"
)
@click.option(
    "--in-memory/--no-in-memory",
    is_flag=True,
    help="Load the dataset in memory"
)
@click.option(
    "--lr",
    type=float,
)
@click.option(
    "--batch-size",
    type=int,
)
@click.option(
    "--num-conv-blocks",
    type=int,
)
@click.option(
    "--fc-hidden-units",
    type=int,
)
def tune_cnn(
    dataset,
    save_best_model,
    num_samples,
    num_epochs,
    force_cpu,
    trial_name,
    ray_dir,
    resize,
    in_memory,
    lr,
    batch_size,
    num_conv_blocks,
    fc_hidden_units
):
    if force_cpu:
        ray.init(num_gpus=0)
        print("--force-cpu flag detected, forcing CPU training by initializing Ray with num_gpus=0")
    # sample only values between around 0.0001 and 0.00001
    config = {
        "lr": lr or tune.choice([0.001, 0.002, 0.003, 0.0001]),
        "batch_size": batch_size or tune.choice([32, 64, 128]),
        "num_conv_blocks": num_conv_blocks or tune.choice([3, 4, 5]),
        "fc_hidden_units": fc_hidden_units or tune.choice([128, 256, 512]),
        "data_dir": dataset,
        "resize": resize,
        "in_memory": in_memory
    }
    scheduler = ASHAScheduler(
        metric="testing_loss",
        mode="min",
        max_t=num_epochs,
        grace_period=min(num_epochs, 5),
        reduction_factor=2
    )

    os.makedirs(save_best_model, exist_ok=True)
    # detect gpus and use if available and not forcing to use cpu
    if torch.cuda.is_available() and not force_cpu:
        trainable = tune.with_resources(
            TuneCNN, resources={"gpu": 1}
        )
    else:
        trainable = TuneCNN
    checkpoint_config = CheckpointConfig(
        checkpoint_frequency=1,
        checkpoint_at_end=True,
    )
    # get the basename of the dataset
    dataset_basename = os.path.basename(dataset)
    default_name = f"cnntune-{dataset_basename}"

    trial_name = trial_name or default_name
    result = tune.run(
        trainable,
        config=config,
        name=trial_name,
        num_samples=num_samples,
        scheduler=scheduler,
        checkpoint_config=checkpoint_config,
        storage_path=ray_dir
    )

    best_trial = result.get_best_trial("testing_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial id: {best_trial.trial_id}")
    print(
        f"Best trial final testing loss: {best_trial.last_result['testing_loss']}")
    print(
        f"Best trial final testing accuracy: {best_trial.last_result['testing_accuracy']}")

    best_checkpoint = result.get_best_checkpoint(
        trial=best_trial, metric="testing_accuracy", mode="max")
    
    best_model_path = os.path.join(save_best_model, f"{trial_name}_best_model.pth")
    best_metadata_path = os.path.join(save_best_model, f"{trial_name}_best_metadata.json")
    print(f"Saving the instance of the best model with highest accuracy to {best_model_path!r}")
    with best_checkpoint.as_directory() as checkpoint_dir:
        model = torch.load(os.path.join(checkpoint_dir, "model.pth"))
        torch.save(model, best_model_path)
        with open(best_metadata_path, "w") as f:
            meta = {
                "trial_id": best_trial.trial_id,
                "config": best_trial.config,
            }
            json.dump(meta, f)



if __name__ == "__main__":
    tune_cnn()