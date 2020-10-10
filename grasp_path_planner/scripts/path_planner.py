# path planner packages
import rospy
import sys
sys.path.append("../../carla_utils/utils")

# ROS Packages
from utility import PathPlan, EnvDesc
from grasp_path_planner.srv import SimService, SimServiceRequest
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
        print("CARLA Global Pose x:",self.prev_env_desc.cur_vehicle_state.location_global.x)
        print("CARLA Global Pose y:", self.prev_env_desc.cur_vehicle_state.location_global.y)
        print("CARLA Global Pose theta:", self.prev_env_desc.cur_vehicle_state.location_global.theta)

        path_plan = self.traj_generator.trajPlan(action, self.prev_env_desc)

        print("Tracking Global Pose x:", path_plan.tracking_pose.x)
        print("Tracking Global Pose y:", path_plan.tracking_pose.y)
        print("Tracking Global Pose theta:", path_plan.tracking_pose.theta)

        req = SimServiceRequest()
        req.path_plan = path_plan
        # import ipdb; ipdb.set_trace()
        self.prev_env_desc = EnvDesc.fromRosMsg(self.sim_service_interface(req).env)
        return self.prev_env_desc, path_plan.end_of_action

    def resetSim(self):
        self.traj_generator.reset()
        reset_msg = PathPlan()
        reset_msg.reset_sim = True
        reset_msg.end_of_action = True
        req = SimServiceRequest()
        req.path_plan = reset_msg
        self.prev_env_desc = EnvDesc.fromRosMsg(self.sim_service_interface(req).env)
        return self.prev_env_desc
