import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import math

import replay_memory
import deep_q_network

import matplotlib
import matplotlib.pyplot as plt

class DQNManager:
    def __init__(self, batch_size, gamma, \
                 eps_start, eps_end, eps_decay, target_update,\
                 input_height, input_width,\
                 capacity, n_actions, device, *args):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.capacity = capacity
        self.n_actions = n_actions
        self.device = device
        self.input_height = input_height
        self.input_width = input_width
        self.memory = None
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.steps_done = 0 #ToDo: Save and Load this Value
        self.loss_graph = [] #ToDo: Make this legit
        self.loss_count = 0

    def initialize(self):
        if self.input_height is not 0:
            self.policy_net = deep_q_network.DQN_Conv(self.input_height, self.input_width, self.n_actions).double().to(self.device)
            self.target_net = deep_q_network.DQN_Conv(self.input_height, self.input_width, self.n_actions).double().to(self.device)
        else:
            # if not a convolutional network
            self.policy_net = deep_q_network.DQN_Linear(self.input_width, self.n_actions).to(self.device)
            self.target_net = deep_q_network.DQN_Linear(self.input_width, self.n_actions, requires_grad=False).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-5)
        #self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = replay_memory.ReplayMemory(self.capacity)

    def saveGraph(self):
        print("SAVE GRAPH")
        x_graph = np.array(range(len(self.loss_graph)))
        loss_g = np.array(self.loss_graph)
        plt.plot(x_graph, loss_g)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig('/home/mayank/Desktop/Loss_Curve.png')

    def selectAction(self, state):

        # greedy eps algorithm
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).view(-1,self.n_actions).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):

        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = replay_memory.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s.view(1,-1) for s in batch.next_state
                                                    if s is not None], dim=0)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).float()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).view(-1,self.n_actions).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        if self.loss_count == 0:
            self.loss_graph.append(loss)
            self.saveGraph()
        self.loss_count += 1
        if self.loss_count == 20:
            self.loss_count = 0

        if self.loss_count % self.target_update == 0:
            self.updateTargetNetwork()
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def newTrainingData(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def exitCondition(self, state_info):
        # ToDo: Exit Condition
        if state_info.reward.time_elapsed > 40:
            return True
        return state_info.reward.collision

    def rewardCalculation(self, state_info):
        # ToDo: Reward Function
        if state_info.reward.collision:
            return -1
        if abs(state_info.cur_vehicle_state.vehicle_location.y - state_info.next_lane.lane[0].pose.y) < 0.05:
            return 1
        return 0

    def saveModel(self, path):
        # ToDo: save both networks
        torch.save(self.target_net.state_dict(), path)
        
    def loadModel(self, path):
        self.target_net.load_state_dict(torch.load(path))

    def updateTargetNetwork(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
