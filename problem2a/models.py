import torch
from typing import List, Tuple
from functools import partial


def get_cnn_image_dimensions(
    image_height: int,
    image_width: int,
    padding: int,
    kernel_size: int,
    stride: int,
    pool_stride: int = 2
) -> Tuple[int, int]:
    """Calcule the new height and width of an image after a convolutional layer.

    Args:
        image_height (int): input image height
        image_width (int): input image width
        padding (int): the padding used for the kernels
        kernel_size (int): the kernel size
        stride (int): the stride applied at the convolutional step
        pool_stride (int, optional): the stride used for any pooling layer. Defaults to 2.

    Returns:
        Tuple[int, int]: new_height, new_width
    """
    new_height = ((image_height - kernel_size + 2*padding) // stride) + 1
    new_width = ((image_width - kernel_size + 2*padding) // stride) + 1
    return new_height//pool_stride, new_width//pool_stride


class CNN2D(torch.nn.Module):

    def __init__(
            self,
            conv1_num_kernels: int,
            conv2_num_kernels: int,
            fc_hidden_units: int,
            image_height: int,
            image_width: int,
            num_classes: int = 5,
            input_channels: int = 1,
            ):
        super(CNN2D, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.conv1, new_dims_after_one = self.convolutional_block(
            in_channels=self.input_channels, out_channels=conv1_num_kernels, kernel_size=3)
        self.conv2, new_dims_after_second = self.convolutional_block(
            in_channels=conv1_num_kernels, out_channels=conv2_num_kernels, kernel_size=3)
        final_h, final_w = new_dims_after_second(
            *new_dims_after_one(image_height=image_height, image_width=image_width))
        output_dim = final_h * final_w * conv2_num_kernels
        self.fc = self.sequential_network(
            input_dim=output_dim, hidden_dim=fc_hidden_units, output_dim=num_classes)
        
    def forward(self, x: torch.Tensor):
        # handle the case where the input is a 3D tensor
        # as this always assumes a batch
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # softmaxed
        return torch.softmax(x, dim=1)
    

    def predict_classes(self, x: torch.Tensor):
        """Use predict to get the predicted class of an input tensor.
        This assumes always batch, so if you want to predict a single image
        it will return a tensor of size 1.
        """
        self.eval()
        predicted_classes = 0
        with torch.no_grad():
            predicted_classes = torch.argmax(self.forward(x), dim=1)
        self.train()
        return predicted_classes
        
    def convolutional_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1
    ) -> Tuple[torch.nn.Module, callable]:
        nn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.3)
        )
        new_dim_calc = partial(get_cnn_image_dimensions,
                               kernel_size=kernel_size, stride=stride, padding=0, pool_stride=2)
        return nn, new_dim_calc
    
    def sequential_network(self, input_dim: int, hidden_dim: int, output_dim: int):
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_dim, output_dim)
        )