import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    h - height of the input image
    w - width of the input image
    outputs = number of action space
    """
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()


        # convolutional layer
        self.conv1 = nn.Conv2d(3,16, kernel_size=5, stride = 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride = 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride = 2)
        self.bn3 = nn.BatchNorm2d(32)

        # setup the network
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size -1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

        # batch for optimization and one element for the next action
        # return: [[action exps], [action exps] ...]
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
