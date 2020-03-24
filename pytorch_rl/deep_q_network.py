import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_Conv(nn.Module):
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

class DQN_Linear(nn.Module):
    def __init__(self, input_size, output_size, requires_grad = True):
        super().__init__()

        # vehicle embedding
        self.vehicle_embedding = nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLu(),
            nn.Linear(32,32)
        )

        # create a neural network
        self.network = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            #nn.Linear(512, 512),
            #nn.ReLU(),
            nn.Linear(512, output_size)
        )
        self.loss_fn = F.mse_loss
        self.optimiser = torch.optim.Adam(self.network.parameters())

        if not requires_grad:
            for parameter in self.network.parameters():
                parameter.requires_grad = False

    def forward(self, X):
        print ("Forward")
        veh1_comb = torch.Tensor(X.shape[0],8,1)
        veh1_comb[:,:4,0] = X[:,:4]
        veh1_comb[:,4:,0] = X[:,4:8]
        veh1_embedding = self.vehicle_embedding(veh1_comb)

        veh2_comb = torch.Tensor(X.shape[0],8,1)
        veh2_comb[:,:4,0] = X[:,:4]
        veh2_comb[:,4:,0] = X[:,8:12]
        veh2_embedding = self.vehicle_embedding(veh2_comb)

        veh3_comb = torch.Tensor(X.shape[0],8,1)
        veh3_comb[:,:4,0] = X[:,:4]
        veh3_comb[:,4:,0] = X[:,12:16]
        veh3_embedding = self.vehicle_embedding(veh3_comb)

        veh4_comb = torch.Tensor(X.shape[0],8,1)
        veh4_comb[:,:4,0] = X[:,:4]
        veh4_comb[:,4:,0] = X[:,16:20]
        veh4_embedding = self.vehicle_embedding(veh4_comb)

        veh5_comb = torch.Tensor(X.shape[0],8,1)
        veh5_comb[:,:4,0] = X[:,:4]
        veh5_comb[:,4:,0] = X[:,20:24]
        veh5_embedding = self.vehicle_embedding(veh5_comb)

        veh_embedding_all = torch.cat((veh1_embedding, veh2_embedding, veh3_embedding, veh4_embedding, veh5_embedding), 2)
        veh_embedding_max = torch.max(veh_embedding_all,2)
        out = self.network(veh_embedding_max)
        return out
