# path planner packages
import math
import rospy
import numpy as np
from grasp_path_planner.srv import SimService, SimServiceResponse, SimServiceRequest
from trajectory_generator import TrajGenerator
# other packages
from settings import *

TRAJ_PARAM = None
if(CURRENT_SCENARIO == Scenario.LANE_CHANGE):
    TRAJ_PARAM = {'look_up_distance' : 0 ,\
        'lane_change_length' : 30,\
        'lane_change_time_constant' : 1.05,\
        'lane_change_time_disc' : 0.4,\
        'action_time_disc' : 0.2,\
        'action_duration' : 0.5,\
        'accelerate_amt' : 5,\
        'decelerate_amt' : 5,\
        'min_speed' : 20
    }
else:
    TRAJ_PARAM = {'look_up_distance' : 0 ,\
        'lane_change_length' : 30,\
        'lane_change_time_constant' : 1.05,\
        'lane_change_time_disc' : 0.4,\
        'action_time_disc' : 0.2,\
        'action_duration' : 0.5,\
        'accelerate_amt' : 5,\
        'decelerate_amt' : 30,\
        'min_speed' : 0
    }

class PathPlannerManager:
    def __init__(self):
        self.traj_generator = TrajGenerator(TRAJ_PARAM)
        self.sim_service_interface = None
        self.prev_env_desc = None


    def initialize(self):
        rospy.init_node(NODE_NAME, anonymous=True)
        rospy.wait_for_service(SIM_SERVICE_NAME)
        self.sim_service_interface = rospy.ServiceProxy(SIM_SERVICE_NAME, SimService)

    def performAction(self, action):
        path_plan = self.traj_generator.trajPlan(action, self.prev_env_desc)
        req = SimServiceRequest()
        req.path_plan = path_plan
        self.prev_env_desc = self.sim_service_interface(req).env
        return self.prev_env_desc, path_plan.end_of_action

    def resetSim(self):
        self.traj_generator.reset()
        reset_msg = PathPlan()
        reset_msg.reset_sim = True
        reset_msg.end_of_action = True
        reset_msg.sent_time = rospy.Time.now()
        req = SimServiceRequest()
        req.path_plan = reset_msg
        self.prev_env_desc = self.sim_service_interface(req).env
        return self.prev_env_desc
