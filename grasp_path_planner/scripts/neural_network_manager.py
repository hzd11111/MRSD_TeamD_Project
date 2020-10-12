import numpy as np
# RL Packages
from stable_baselines import DQN

# other packages
from settings import Scenario, RLDecision


class NNManager:
    """
    A neural network manager class to perform inference
    """
    def __init__(self, event: Scenario):
        """
        Class constructor.
        Handles the neural_network inference based on the scenario
        """
        self.event = event

    def initialize(self, model_path: str):
        """
        loads the model
        Args:
        :param model_path: (str) the path to the model
        """
        self.neural_network = DQN.load(model_path)

    def makeDecision(self, env_embedding: np.ndarray) -> RLDecision:
        """
        Makes a decision using the neural network
        Args:
        :param env_embedding: Embedding of environment information
        """
        action, _ = self.neural_network.predict(env_embedding)
        return RLDecision(action)


class NeuralNetworkSelector:
    """
    TODO: Selects the neural network based on what?
    """
    def __init__(self):
        pass
