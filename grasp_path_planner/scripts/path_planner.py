# path planner packages
import math
import rospy
import numpy as np
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import RLCommand
from grasp_path_planner.msg import Pedestrian
from grasp_path_planner.msg import PathPlan
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
    

class VecTemp:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def add(self, other):
        return VecTemp(self.x + other.x, self.y + other.y)

    def sub(self, other):
        return VecTemp(self.x - other.x, self.y - other.y)

    def dot(self, other):
        upper = self.x * other.x + self.y * other.y
        lower = self.norm() * other.norm()
        if lower <= 0.00001:
            return 1
        return upper / lower


class PoseTemp:
    def __init__(self, ros_pose=False):
        if ros_pose:
            self.x = ros_pose.x
            self.y = ros_pose.y
            self.theta = self.wrapToPi(ros_pose.theta)
        else:
            self.x = 0
            self.y = 0
            self.theta = 0

    def wrapToPi(self, theta):
        return (theta + math.pi) % (2. * math.pi) - math.pi

    def distance(self, pose):
        return math.sqrt((self.x - pose.x) ** 2. + (self.y - pose.y) ** 2.)

    def add(self, pose):
        new_pose = PoseTemp()
        new_pose.x = self.x + pose.x * math.cos(self.theta) - pose.y * math.sin(self.theta)
        new_pose.y = self.y + pose.x * math.sin(self.theta) + pose.y * math.cos(self.theta)
        new_pose.theta = self.wrapToPi(self.theta + pose.theta)
        return new_pose

    def vecTo(self, pose):
        new_vec = VecTemp()
        new_vec.x = pose.x - self.x
        new_vec.y = pose.y - self.y
        return new_vec

    def vecFromTheta(self):
        return VecTemp(math.cos(self.theta), math.sin(self.theta))

    def isInfrontOf(self, pose):
        diff_vec = pose.vecTo(self)
        other_vec = pose.vecFromTheta()
        return diff_vec.dot(other_vec) > 0

    def scalarMultiply(self, scalar):
        new_pose = PoseTemp()
        new_pose.x = self.x * scalar
        new_pose.y = self.y * scalar
        new_pose.theta = self.theta * scalar


class PoseSpeedTemp(PoseTemp):
    def __init__(self):
        PoseTemp.__init__(self)
        self.speed = 0

    def addToPose(self, pose):
        new_pose = PoseSpeedTemp()
        new_pose.x = pose.x + self.x * math.cos(pose.theta) - self.y * math.sin(pose.theta)
        new_pose.y = pose.y + self.x * math.sin(pose.theta) + self.y * math.cos(pose.theta)
        new_pose.theta = self.wrapToPi(self.theta + pose.theta)
        new_pose.speed = self.speed
        return new_pose

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
