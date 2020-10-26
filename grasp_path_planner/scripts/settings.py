import os
from enum import Enum
from options import Scenario

dir_path = os.path.dirname(os.path.realpath(__file__))


class Mode(Enum):
    TRAIN = 0
    TEST = 1


############ Mode and Model Selection ##############################
CURRENT_SCENARIO = Scenario.LEFT_TURN
CURRENT_MODE = Mode.TEST

if CURRENT_SCENARIO == Scenario.LANE_FOLLOWING:
    MODEL_SAVE_PATH = dir_path + "/Models/DQN_Lane_Following"
    MODEL_LOAD_PATH = dir_path + "/Models/DQN_Model_Lane_Following"
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

################## Test Mode Arguments ######################
if CURRENT_SCENARIO == Scenario.LANE_FOLLOWING:
    TOWN_ID = "Town01"

    ROAD_IDs = [12]
    LEFT_LANE_ID = 4  # Not required for this scenario
    RIGHT_LANE_ID = -1

    EGO_INIT_SPEED = 48.5  # 46.5 for road 40, #44.5 for road 37
    NPC_INIT_SPEED = 20  # Not required for this scenario

    EGO_SPAWN_IDX = 80
    EGO_VEHICLE_MAKE = "tt"

    WALKER_MAX_SPEED = 1.2
    WALKER_SAMPLE_SPEED = True
    WALKER_SPAWN_DIST_MIN = 16
    WALKER_SPAWN_DIST_MAX = 25

    NPC_SPAWN_POINT_GAP_LOW = 15  # Not required for this scenario
    NPC_SPAWN_POINT_GAP_HIGH = 30  # Not required for this scenario
    LIVES_MATTER = False  # Not required for this scenario

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
