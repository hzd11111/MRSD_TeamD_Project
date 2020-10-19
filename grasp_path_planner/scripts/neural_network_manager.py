import numpy as np
# RL Packages
from stable_baselines import DQN

# other packages
from state_manager import StateManager
from RLManager import GeneralRLManager
from options import Scenario, RLDecision


class NNManager:
    """
    A neural network manager class to perform inference
    """
    def __init__(self):
        """
        Class constructor.
        Handles the neural_network inference based on the scenario
        """
        self.state_manager = StateManager()
        self.decision_maker = GeneralRLManager()
        self.neural_nets = {}

    def initialize(self, model_paths):
        """
        loads the model
        Args:
        :param model_paths: dict (Scenario Enu,path) the path to the model
        """
        for key in model_paths:
            self.neural_nets[key] = DQN.load(model_paths[key])

    def makeDecision(self, env_desc, scenario) -> RLDecision:
        """
        Makes a decision using the neural network
        Args:
        :param env_embedding: Embedding of environment information
        """
        state = self.state_manager.embedState(env_desc, scenario)
        action, _ = self.neural_network.predict(env_embedding)
        return self.decision_maker.convertDecision(action, scenario)


class NeuralNetworkSelector:
    """
    TODO: Selects the neural network based on what?
    """
    def __init__(self):
        pass
