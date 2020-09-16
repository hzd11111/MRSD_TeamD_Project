# RL Packages
from stable_baselines import DQN

# ROS Packages
from grasp_path_planner.msg import EnvironmentState

# other packages
from state_manager import StateManager
from settings import Scenario, RLDecision, CONVERT_TO_LOCAL


class NNManager:
    """
    A neural network manager class to perform inference
    """
    def __init__(self, event: Scenario):
        """
        Class constructor.
        Handles the neural_network inference based on the scenario
        """
        self.state_manager = StateManager()
        self.event = event

    def initialize(self, model_path: str):
        """
        loads the model
        Args:
        :param model_path: (str) the path to the model
        """
        self.neural_network = DQN.load(model_path)

    def makeDecision(self, env_desc: EnvironmentState) -> RLDecision:
        """
        Makes a decision using the neural network
        Args:
        :param env_desc: (EnvironmentState) A ROS Message describing the environment
        """
        env_state = self.state_manager.embedState(env_desc, self.event, CONVERT_TO_LOCAL)
        action, _ = self.neural_network.predict(env_state)
        return RLDecision(action)


class NeuralNetworkSelector:
    """
    TODO: Selects the neural network based on what?
    """
    def __init__(self):
        pass
