from enum import Enum, unique


@unique
class RLDecision(Enum):
    CONSTANT_SPEED = 0
    ACCELERATE = 1
    DECELERATE = 2
    SWITCH_LANE_LEFT = 3
    SWITCH_LANE_RIGHT = 4
    LEFT_TURN = 5
    RIGHT_TURN = 6
    GO_STRAIGHT = 7
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
    LANE_CHANGE = 0
    PEDESTRIAN = 1
    LEFT_TURN = 2
    GO_STRAIGHT = 3
    RIGHT_TURN = 4


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
