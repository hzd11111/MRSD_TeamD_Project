from enum import Enum, unique


@unique
class RLDecision(Enum):
    CONSTANT_SPEED = 0
    ACCELERATE = 1
    DECELERATE = 2
    SWITCH_LANE_LEFT = 3
    SWITCH_LANE_RIGHT = 4
    GLOBAL_PATH_CONSTANT_SPEED = 5
    GLOBAL_PATH_ACCELERATE = 6
    GLOBAL_PATH_DECELERATE = 7
    STOP = 8
    NO_ACTION = 9


@unique
class PedestrainPriority(Enum):
    GREEN, REF, JAYWALK = range(3)


@unique
class TrafficLightStatus(Enum):
    GREEN, YELLOW, RED = range(3)


@unique
class Scenario(Enum):
    SWITCH_LANE_LEFT = 0
    SWITCH_LANE_RIGHT = 1
    LANE_FOLLOWING = 2
    LEFT_TURN = 3
    GO_STRAIGHT = 4
    RIGHT_TURN = 5
    PEDESTRIAN = 6  # NEED TO REMOVE THIS. KEPT IT BECAUSE SCENARIO MANAGER IS USING IT
    P2P = 7
    DONE = 8
    STOP = 9


@unique
class GlobalPathAction(Enum):
    NO_ACTION = 0
    SWITCH_LANE_LEFT = 1
    SWITCH_LANE_RIGHT = 2
    LEFT_TURN = 3
    RIGHT_TURN = 4
    GO_STRAIGHT = 5
    STOP = 6


@unique
class StopLineStatus(Enum):
    NO_STOP = 0
    STRAIGHT_STOP = 1
    LEFT_TURN_STOP = 2
    RIGHT_TURN_STOP = 3
