import os
from enum import Enum
from options import Scenario
import carla
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


class Mode(Enum):
    TRAIN = 0
    TEST = 1


############ Mode and Model Selection ##############################
CURRENT_SCENARIO = Scenario.GO_STRAIGHT
CURRENT_MODE = Mode.TRAIN
WANDB_DRYRUN = True
VIZ = False

NEW_RUN = True
#assert !(CURRENT_SCENARIO==Scenario.P2P and CURRENT_MODE==Mode.TRAIN), "P2P Cannot be called in train mode"

if CURRENT_SCENARIO == Scenario.LANE_FOLLOWING:
    MODEL_SAVE_PATH = dir_path + "/Models/DQN_Lane_Following"
    MODEL_LOAD_PATH = dir_path + "/Models/DQN_Lane_Following"
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
    MODEL_LOAD_PATH = dir_path + "/Models/DQN_Straight"
    MODEL_CP_PATH = dir_path + "/Models/Straight_CP"
elif CURRENT_SCENARIO == Scenario.RIGHT_TURN:
    MODEL_SAVE_PATH = dir_path + "/Models/DQN_Right_Turn"
    MODEL_LOAD_PATH = dir_path + "/Models/DQN_Right_Turn"
    MODEL_CP_PATH = dir_path + "/Models/Right_Turn_CP"
elif CURRENT_SCENARIO == Scenario.LEFT_TURN:
    MODEL_SAVE_PATH = dir_path + "/Models/DQN_Left_Turn"
    MODEL_LOAD_PATH = dir_path + "/Models/DQN_Left_Turn"
    MODEL_CP_PATH = dir_path + "/Models/Left_Turn_CP"

INTERSECTION_SCENARIOS = [Scenario.GO_STRAIGHT, Scenario.RIGHT_TURN, 
                                                Scenario.LEFT_TURN]
LANE_SCENARIOS = [Scenario.LANE_FOLLOWING, Scenario.SWITCH_LANE_RIGHT,
                                        Scenario.SWITCH_LANE_LEFT]

################ CARLA Settings Arguments ##################
MAP = 'Town05'
WEATHER = 'ClearNoon'
SYNCHRONOUS = True
FIXED_DELTA_SECONDS = 0.05
NO_RENDER_MODE = False

# TRAFFIC MANAGER SETTINGS
TM_PORT = 8000
AUTO_LANE_CHANGE = None
DISTANCE_TO_LEADING_VEHICLES = 2
TARGET_SPEED = 30
HYBRID_PHYSICS_MODE = None 
HYBRID_PHYSICS_RADIUS = 20
IGNORE_LIGHTS_PERCENTAGE = 100
IGNORE_SIGNS_PERCENTAGE = 100
ACTOR_SIMULATE_PHYSICS = False

# EGO VEHICLE PROPS
EGO_VEHICLE_MODEL = 'vehicle.tesla.model3'
NON_EGO_VEHICLE_MODEL = 'vehicle.mustang.mustang'


################## Global Scenario Params #################
DISTANCE_TO_INTERSECTION_FOR_SCENARIO_CHANGE = 20
STOP_LINE_DISTANCE_FOR_LANE_CHANGE_TERMINATE = 5


################## Test Mode Arguments ######################
if CURRENT_SCENARIO == Scenario.LANE_FOLLOWING:

    # add town params? or assume we are only working with Town05
    TOWN_ID = "Town05"

    SPAWN_LOCS_VEC = [(53, 205, 0.1)] #list of potential spawn points, randomize?
    
    LANE_FOLLOWING_CONFIG = {
                    "distance_bwn_waypoints":1,
                    "target_speed":15,
                    "warm_start":True,
                    "warm_start_duration":2,
                    # configure non-ego veh count
                    "min_non_ego_veh":1,
                    "max_non_ego_veh":6, # also update max_q_pos
                    # distance between variables
                    "max_dist_bwn_veh":15,
                    "min_dist_bwn_veh":3,
                    "average_car_length":5,
                    # scenario variables
                    "goal_distance_to_travel":30,          
                    }
    spectator_trans = carla.Transform(carla.Location(x=53, y=205, z=50), \
                                    carla.Rotation(pitch=-39, yaw=41, roll=0))


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
