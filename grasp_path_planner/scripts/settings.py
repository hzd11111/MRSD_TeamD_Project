from enum import Enum


class Scenario(Enum):
    LANE_CHANGE = 0
    PEDESTRIAN = 1

# class RLDecision(Enum):
#     CONSTANT_SPEED = 0
#     ACCELERATE = 2
#     DECELERATE = 3
#     SWITCH_LANE = 1
#     NO_ACTION = 4

# TODO: Currently all the decisions are under one enum. If this is so, we cannot directly
# convert the argmax of the neural network into an enum as each network might not have the same
# number of heads. We will need a more sophisticated enum


class RLDecision(Enum):
    CONSTANT_SPEED = 0
    ACCELERATE = 1
    DECELERATE = 2
    SWITCH_LANE = 3
    NO_ACTION = 4


N_DISCRETE_ACTIONS = 4
CONVERT_TO_LOCAL = True
SIM_SERVICE_NAME = "simulator"
NODE_NAME = "full_grasp_planner"
INVERT_ANGLES = True
OLD_REWARD = False
