import os
from enum import Enum
dir_path = os.path.dirname(os.path.realpath(__file__))

class Scenario(Enum):
    LANE_CHANGE=0
    PEDESTRIAN=1

class Mode(Enum):
    TRAIN=0
    TEST=1


    
    
    

############ Mode and Model Selection ##############################
CURRENT_SCENARIO = Scenario.PEDESTRIAN
CURRENT_MODE = Mode.TEST

if(CURRENT_SCENARIO == Scenario.PEDESTRIAN):
    MODEL_SAVE_PATH = dir_path + "/Models/DQN_Model_CARLA_Ped"
    MODEL_LOAD_PATH = dir_path + "/Models/DQN_Model_CARLA_Ped5"
else:
    MODEL_SAVE_PATH = dir_path + "/Models/DQN_LANE_SWITCH"
    MODEL_LOAD_PATH = dir_path + "/DQN_20min"
    

################## Test Mode Arguments ######################
if(CURRENT_SCENARIO == Scenario.PEDESTRIAN):
    TOWN_ID = "Town04"
    
    ROAD_IDs = [40]
    LEFT_LANE_ID = 3 #Not required for this scenario
    RIGHT_LANE_ID = 4
    
    EGO_INIT_SPEED = 28
    NPC_INIT_SPEED = 20 #Not required for this scenario
    
    EGO_SPAWN_IDX = 100
    EGO_VEHICLE_MAKE = "model3"
    
    WALKER_MAX_SPEED = 1.4
    WALKER_SAMPLE_SPEED = True
    WALKER_SPAWN_DIST_MIN = 16
    WALKER_SPAWN_DIST_MAX = 18
    
    NPC_SPAWN_POINT_GAP_LOW = 15 #Not required for this scenario
    NPC_SPAWN_POINT_GAP_HIGH = 30 #Not required for this scenario
    LIVES_MATTER = False #Not required for this scenario

else:
    
    TOWN_ID = "Town04"
    
    ROAD_IDs = [40]
    LEFT_LANE_ID = 3
    RIGHT_LANE_ID = 4
    
    EGO_INIT_SPEED = 28
    NPC_INIT_SPEED = 20
    EGO_SPAWN_IDX = 100
    EGO_VEHICLE_MAKE = "model3"   
    
    LOW_NUM_VEHICLES = 8
    HIGH_NUM_VEHICLES = 12
    
    NPC_SPAWN_POINT_GAP_LOW = 15
    NPC_SPAWN_POINT_GAP_HIGH = 30
    
    LIVES_MATTER = False
    


















N_DISCRETE_ACTIONS = 4
CONVERT_TO_LOCAL = True
SIM_SERVICE_NAME = "simulator"
NODE_NAME = "full_grasp_planner"
INVERT_ANGLES=True
OLD_REWARD=None


if(CURRENT_SCENARIO == Scenario.PEDESTRIAN):
    class RLDecision(Enum):
        CONSTANT_SPEED = 0
        ACCELERATE = 1
        DECELERATE = 2
        SWITCH_LANE = 3
        NO_ACTION = 4
        
    OLD_REWARD=False
    
else:
    class RLDecision(Enum):
        CONSTANT_SPEED = 0
        ACCELERATE = 2
        DECELERATE = 3
        SWITCH_LANE = 1
        NO_ACTION = 4
        
    OLD_REWARD=True
        
    
###### INIT TO ACTUAL SPEED MAPPINGS FOR EGO VEHICLE ######
# Map1 : TT : 42.8 for 32 kmph
#
# Town04 : Model3 : Road 40: 
# 20 -> 18Kmph
# 30 -> 