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
            num_conv_blocks: int,
            fc_hidden_units: int,
            image_height: int,
            image_width: int,
            num_classes: int = 5,
            input_channels: int = 1,
            ):
        super(CNN2D, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.total_conv_blocks = num_conv_blocks
        self._current_input_channels = input_channels
        initial_num_kernels = 64
        self.convs, flattened_dim_cal = self.build_convolutional_layers(
            initial_num_kernels)
        output_dim = flattened_dim_cal(image_height, image_width)
        self.fc = self.sequential_network(
            input_dim=output_dim, hidden_dim=fc_hidden_units, output_dim=num_classes)

    def build_convolutional_layers(self, initial_num_kernels: int):
        # make convolutional blocks by doing two convolutional layers
        # followed by max pool, and repeat
        layers = []
        funcs = []
        for _ in range(self.total_conv_blocks):
            block, func = self.convolutional_block(
                in_channels=self._current_input_channels,
                out_channels=initial_num_kernels,
                kernel_size=3
            )
            self._current_input_channels = initial_num_kernels
            initial_num_kernels *= 2
            layers.append(block)
            funcs.append(func)
        def flattened_dim_cal(h, w):
            for f in funcs:
                h, w = f(h, w)
            return h*w*self._current_input_channels
        return torch.nn.Sequential(*layers), flattened_dim_cal
        
    def forward(self, x: torch.Tensor):
        # handle the case where the input is a 3D tensor
        # as this always assumes a batch
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.convs(x)
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
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.3)
        )
        new_dim_calc = partial(get_cnn_image_dimensions,
                               kernel_size=kernel_size, stride=stride, padding=0, pool_stride=2)
        return nn, new_dim_calc
    
    def sequential_network(self, input_dim: int, hidden_dim: int, output_dim: int):
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim*2, hidden_dim*4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim*4, output_dim)
        )