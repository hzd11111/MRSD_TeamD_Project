import os
from enum import Enum
from options import Scenario
import carla
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


class Mode(Enum):
    TRAIN = 0
    TEST = 1


##### TUNABLE SETTINGS #####
NUM_NON_EGO_VEHICLES = 200
TEST_ROUTE = 2
IGNORE_LIGHTS_PERCENTAGE = 0
TARGET_SPEED = 20

CONTROL_TRAFFIC_LIGHTS = False
# TRAFFIC LIGHT DURATION
TRAFFIC_LIGHT_RED_DURATION = 2
TRAFFIC_LIGHT_GREEN_DURATION = 5
TRAFFIC_LIGHT_YELLOW_DURATION = 0.25
#############################

############ Mode and Model Selection ##############################
CURRENT_SCENARIO = Scenario.P2P
CURRENT_MODE = Mode.TEST
WANDB_DRYRUN = True
NEW_RUN = False
DEBUG = True

VIZ = False
VIS_LIDAR = False


P2P_LOOPING = False
SPECTATOR_FOLLOWING = True


if CURRENT_SCENARIO == Scenario.LANE_FOLLOWING:
    MODEL_SAVE_PATH = dir_path + "/Models/DQN_Lane_Following"
    # MODEL_LOAD_PATH = dir_path + "/Models/DQN_Lane_Following"
    MODEL_LOAD_PATH = dir_path + "/Models/DQN_Lane_Following_Extended"
    # MODEL_LOAD_PATH = dir_path + "/Models/DQN_Lane_Following_model_18000_steps.zip"
    MODEL_CP_PATH = dir_path + "/Models/Lane_Following_CP"
elif CURRENT_SCENARIO == Scenario.SWITCH_LANE_LEFT:
    MODEL_SAVE_PATH = dir_path + "/Models/DQN_Lane_Switch_Left"
    MODEL_LOAD_PATH = dir_path + "/Models/DQN_Lane_Switch_Left"
    MODEL_CP_PATH = dir_path + "/Models/DQN_Lane_Switch_Left_CP"
elif CURRENT_SCENARIO == Scenario.SWITCH_LANE_RIGHT:
    MODEL_SAVE_PATH = dir_path + "/Models/DQN_Lane_Switch_Right"
    MODEL_LOAD_PATH = dir_path + "/Models/DQN_Lane_Switch_Right"
    MODEL_CP_PATH = dir_path + "/Models/Lane_Switch_Right_CP"
elif CURRENT_SCENARIO == Scenario.GO_STRAIGHT:
    MODEL_SAVE_PATH = dir_path + "/Models/DQN_Straight"
    MODEL_LOAD_PATH = dir_path + "/Models/DQN_Straight_Extended"
    MODEL_CP_PATH = dir_path + "/Models/Straight_CP"
elif CURRENT_SCENARIO == Scenario.RIGHT_TURN:
    MODEL_SAVE_PATH = dir_path + "/Models/DQN_Right_Turn"
    MODEL_LOAD_PATH = dir_path + "/Models/DQN_Right_Turn"
    MODEL_CP_PATH = dir_path + "/Models/Right_Turn_CP"
elif CURRENT_SCENARIO == Scenario.LEFT_TURN:
    MODEL_SAVE_PATH = dir_path + "/Models/DQN_Left_Turn"
    MODEL_LOAD_PATH = dir_path + "/Models/DQN_Left_Turn"
    MODEL_CP_PATH = dir_path + "/Models/Left_Turn_CP_3"

INTERSECTION_SCENARIOS = [Scenario.GO_STRAIGHT, Scenario.RIGHT_TURN, Scenario.LEFT_TURN]
LANE_SCENARIOS = [
    Scenario.LANE_FOLLOWING,
    Scenario.SWITCH_LANE_RIGHT,
    Scenario.SWITCH_LANE_LEFT,
]

################ CARLA Settings Arguments ##################
MAP = "Town05"
WEATHER = "ClearNoon"
SYNCHRONOUS = True
FIXED_DELTA_SECONDS = 0.05
NO_RENDER_MODE = False


# TRAFFIC MANAGER SETTINGS
TM_PORT = 8000
AUTO_LANE_CHANGE = None
DISTANCE_TO_LEADING_VEHICLES = 2
HYBRID_PHYSICS_MODE = None
HYBRID_PHYSICS_RADIUS = 20
IGNORE_SIGNS_PERCENTAGE = 0
ACTOR_SIMULATE_PHYSICS = False

# EGO VEHICLE PROPS
EGO_VEHICLE_MODEL = "vehicle.tesla.model3"
NON_EGO_VEHICLE_MODEL = "vehicle.mustang.mustang"


################## Global Scenario Params #################
DISTANCE_TO_INTERSECTION_FOR_SCENARIO_CHANGE = 20
if (
    CURRENT_SCENARIO == Scenario.GO_STRAIGHT
    or CURRENT_SCENARIO == Scenario.LEFT_TURN
    or CURRENT_SCENARIO == Scenario.RIGHT_TURN
):
    STOP_LINE_DISTANCE_FOR_LANE_CHANGE_TERMINATE = 2
elif (
    CURRENT_SCENARIO == Scenario.SWITCH_LANE_LEFT
    or CURRENT_SCENARIO == Scenario.SWITCH_LANE_RIGHT
):
    STOP_LINE_DISTANCE_FOR_LANE_CHANGE_TERMINATE = 8
else:
    STOP_LINE_DISTANCE_FOR_LANE_CHANGE_TERMINATE = 4


################## Test Mode Arguments ######################s
if CURRENT_SCENARIO == Scenario.LANE_FOLLOWING:

    # add town params? or assume we are only working with Town05
    TOWN_ID = "Town05"

    SPAWN_LOCS_VEC = [  # (53, 205, 0.1), # highway
        (-48.5, -106, 0.1),  # curved road
        (-66, -95, 0.1),  # short straight road 1
        (-144, -92, 0.1),  # super short road
        (-51.5, -74.5, 0.1),  # medium length road
        (-64, -4.3, 0.1),  # medium length road
    ]  # list of potential spawn points, randomize?

    LANE_FOLLOWING_CONFIG = {
        "road_ids": [6, 7, 45, 46, 8],
        "distance_bwn_waypoints": 1,
        "target_speed": 5,
        "warm_start": True,
        "warm_start_duration": 0.2,
        # configure non-ego veh count
        "min_non_ego_veh": 0,
        "max_non_ego_veh": 4,
        # distance between variables
        "max_dist_bwn_veh": 10,
        "min_dist_bwn_veh": 2,
        "average_car_length": 5,
        "min_dist_to_end_of_lane_from_first_veh": 10
        # scenario variables
        # "goal_distance_to_travel":30,
    }
    spectator_trans = carla.Transform(
        carla.Location(x=53, y=205, z=50), carla.Rotation(pitch=-39, yaw=41, roll=0)
    )


elif CURRENT_SCENARIO == Scenario.SWITCH_LANE_LEFT:
    TOWN_ID = "Town05"

    ROAD_IDs = [37]
    LEFT_LANE_ID = -2
    RIGHT_LANE_ID = -3

    EGO_INIT_SPEED = 30  # 30 for high speed on road40
    NPC_INIT_SPEED = 20
    EGO_SPAWN_IDX = 470
    EGO_VEHICLE_MAKE = "model3"

    LOW_NUM_VEHICLES = 30
    HIGH_NUM_VEHICLES = 35

    NPC_SPAWN_POINT_GAP_LOW = 15
    NPC_SPAWN_POINT_GAP_HIGH = 30

    LIVES_MATTER = False


# TODO: Currently all the decisions are under one enum. If this is so, we cannot directly
# convert the argmax of the neural network into an enum as each network might not have the same
# number of heads. We will need a more sophisticated enum


SIM_SERVICE_NAME = "simulator"
NODE_NAME = "full_grasp_planner"


###### INIT TO ACTUAL SPEED MAPPINGS FOR EGO VEHICLE ######
# Map1 : TT : 42.8 for 32 kmph
#
# Town04 : Model3 : Road 40:
# 20 -> 18Kmph
# 30 ->
