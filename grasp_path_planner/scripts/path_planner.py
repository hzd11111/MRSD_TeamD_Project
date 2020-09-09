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
# other packages
from settings import *

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

class TrajGenerator:
    SAME_POSE_THRESHOLD = 2
    SAME_POSE_LOWER_THRESHOLD = 0.02

    def __init__(self, traj_parameters):
        self.traj_parameters = traj_parameters
        self.current_action = RLDecision.NO_ACTION
        self.generated_path = []
        self.path_pointer = 0
        self.action_start_time = 0
        self.start_speed = 0
        self.reset()

    def reset(self, cur_act=RLDecision.NO_ACTION, start_speed=0):
        self.current_action = cur_act
        self.generated_path = []
        self.path_pointer = 0
        self.action_start_time = 0
        self.start_speed = start_speed

    def trajPlan(self, rl_data, sim_data):
        if len(sim_data.current_lane.lane) <= 0 or len(sim_data.next_lane.lane) <= 0:
            print("Lane has no component")
        if self.current_action == RLDecision.SWITCH_LANE:
            return self.laneChangeTraj(sim_data)
        elif self.current_action == RLDecision.ACCELERATE:
            return self.accelerateTraj(sim_data)
        elif self.current_action == RLDecision.DECELERATE:
            return self.decelerateTraj(sim_data)
        elif self.current_action == RLDecision.CONSTANT_SPEED:
            return self.constSpeedTraj(sim_data)
        elif rl_data == RLDecision.CONSTANT_SPEED:
            return self.constSpeedTraj(sim_data)
        elif rl_data == RLDecision.ACCELERATE:
            new_path_plan = self.accelerateTraj(sim_data)
            return new_path_plan
        elif rl_data == RLDecision.DECELERATE:
            new_path_plan = self.decelerateTraj(sim_data)
            return new_path_plan
        elif rl_data == RLDecision.SWITCH_LANE:
            return self.laneChangeTraj(sim_data)
        else:
            print("RL Decision Error")

    def findNextLanePose(self, cur_vehicle_pose, lane_pose_array):
        closest_pose = lane_pose_array[0]

        for lane_waypoint in lane_pose_array:
            closest_pose = lane_waypoint
            way_pose = PoseTemp(lane_waypoint.pose)
            if way_pose.distance(cur_vehicle_pose) < TrajGenerator.SAME_POSE_THRESHOLD and \
                    way_pose.isInfrontOf(cur_vehicle_pose) and \
                    way_pose.distance(cur_vehicle_pose) > TrajGenerator.SAME_POSE_LOWER_THRESHOLD:
                return closest_pose
        return closest_pose

    def constSpeedTraj(self, sim_data):
        # duration evaluation
        if not self.current_action == RLDecision.CONSTANT_SPEED:
            self.reset(RLDecision.CONSTANT_SPEED, sim_data.cur_vehicle_state.vehicle_speed)

            # set action start time
            self.action_start_time = sim_data.reward.time_elapsed
        new_sim_time = sim_data.reward.time_elapsed
        # current vehicle state
        cur_vehicle_state = sim_data.cur_vehicle_state
        cur_vehicle_pose = PoseTemp(cur_vehicle_state.vehicle_location)
        cur_vehicle_speed = cur_vehicle_state.vehicle_speed

        # determine the closest next pose in the current lane
        lane_pose_array = sim_data.current_lane.lane
        closest_pose = self.findNextLanePose(cur_vehicle_pose, lane_pose_array)

        end_of_action = False
        #if rl_data.reset_run:
        #    end_of_action = True

        action_progress = (new_sim_time - self.action_start_time) / self.traj_parameters['action_duration']

        if action_progress >= 1.:
            end_of_action = True
            action_progress = 1.

        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = closest_pose.pose
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = max(self.traj_parameters['min_speed'],self.start_speed)
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = False

        # add future poses
        new_path_plan.future_poses = []
        closest_pose_temp = PoseTemp(closest_pose.pose)
        for ts in np.arange(self.traj_parameters['action_time_disc'], \
                self.traj_parameters['action_duration'] - (new_sim_time - self.action_start_time), \
                            self.traj_parameters['action_time_disc']):
            delta_pose = PoseTemp()
            delta_pose.x = ts * self.start_speed / 3.6
            new_pose = closest_pose_temp.add(delta_pose)
            new_pose_ros = Pose2D()
            new_pose_ros.x = new_pose.x
            new_pose_ros.y = new_pose.y
            new_pose_ros.theta = new_pose.theta
            new_path_plan.future_poses.append(new_pose_ros)

        if end_of_action:
            self.reset()

        return new_path_plan

    def accelerateTraj(self, sim_data):
        # duration evaluation
        if not self.current_action == RLDecision.ACCELERATE:
            self.reset(RLDecision.ACCELERATE, sim_data.cur_vehicle_state.vehicle_speed)

            # set action start time
            self.action_start_time = sim_data.reward.time_elapsed
        new_sim_time = sim_data.reward.time_elapsed
        # current vehicle state
        cur_vehicle_state = sim_data.cur_vehicle_state
        cur_vehicle_pose = PoseTemp(cur_vehicle_state.vehicle_location)
        cur_vehicle_speed = cur_vehicle_state.vehicle_speed

        # determine the closest next pose in the current lane
        lane_pose_array = sim_data.current_lane.lane
        closest_pose = self.findNextLanePose(cur_vehicle_pose, lane_pose_array)

        end_of_action = False
        #if rl_data.reset_run:
        #    end_of_action = True

        action_progress = (new_sim_time - self.action_start_time) / self.traj_parameters['action_duration']

        if action_progress >= 1.:
            end_of_action = True
            action_progress = 1.

        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = closest_pose.pose
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = max(self.traj_parameters['min_speed'],self.start_speed + action_progress * self.traj_parameters['accelerate_amt'])
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = False

        # add future poses
        new_path_plan.future_poses = []
        closest_pose_temp = PoseTemp(closest_pose.pose)
        acc_per_sec = self.traj_parameters['accelerate_amt'] / self.traj_parameters['action_duration']
        for ts in np.arange(self.traj_parameters['action_time_disc'], \
                self.traj_parameters['action_duration'] - (new_sim_time - self.action_start_time), \
                            self.traj_parameters['action_time_disc']):
            delta_pose = PoseTemp()
            delta_pose.x = (ts * (self.start_speed + action_progress * self.traj_parameters['accelerate_amt']) +\
                            acc_per_sec * (ts**2.) / 2.) / 3.6
            new_pose = closest_pose_temp.add(delta_pose)
            new_pose_ros = Pose2D()
            new_pose_ros.x = new_pose.x
            new_pose_ros.y = new_pose.y
            new_pose_ros.theta = new_pose.theta
            new_path_plan.future_poses.append(new_pose_ros)

        if end_of_action:
            self.reset()
        return new_path_plan

    def decelerateTraj(self, sim_data):
        # duration evaluation
        if not self.current_action == RLDecision.DECELERATE:
            self.reset(RLDecision.DECELERATE, sim_data.cur_vehicle_state.vehicle_speed)

            # set action start time
            self.action_start_time = sim_data.reward.time_elapsed
        new_sim_time = sim_data.reward.time_elapsed
        # current vehicle state
        cur_vehicle_state = sim_data.cur_vehicle_state
        cur_vehicle_pose = PoseTemp(cur_vehicle_state.vehicle_location)
        cur_vehicle_speed = cur_vehicle_state.vehicle_speed

        # determine the closest next pose in the current lane
        lane_pose_array = sim_data.current_lane.lane
        closest_pose = self.findNextLanePose(cur_vehicle_pose, lane_pose_array)

        end_of_action = False
        #if rl_data.reset_run:
        #    end_of_action = True

        action_progress = (new_sim_time - self.action_start_time) / self.traj_parameters['action_duration']

        if action_progress >= 1.:
            end_of_action = True
            action_progress = 1.

        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = closest_pose.pose
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = max(self.traj_parameters['min_speed'],
                                           self.start_speed - action_progress * self.traj_parameters['decelerate_amt'])
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = False

        # add future poses
        new_path_plan.future_poses = []
        closest_pose_temp = PoseTemp(closest_pose.pose)
        dec_per_sec = self.traj_parameters['decelerate_amt'] / self.traj_parameters['action_duration']
        for ts in np.arange(self.traj_parameters['action_time_disc'], \
                self.traj_parameters['action_duration'] - (new_sim_time - self.action_start_time), \
                            self.traj_parameters['action_time_disc']):
            delta_pose = PoseTemp()
            delta_pose.x = (ts * (self.start_speed - action_progress * self.traj_parameters['accelerate_amt']) -\
                            dec_per_sec * (ts**2.) / 2.) / 3.6
            new_pose = closest_pose_temp.add(delta_pose)
            new_pose_ros = Pose2D()
            new_pose_ros.x = new_pose.x
            new_pose_ros.y = new_pose.y
            new_pose_ros.theta = new_pose.theta
            new_path_plan.future_poses.append(new_pose_ros)

        if end_of_action:
            self.reset()

        return new_path_plan

    def cubicSplineGen(self, cur_lane_width, next_lane_width, v_cur):
        v_cur = v_cur / 3.6
        if v_cur < 5:
            v_cur = 5
        # determine external parameters
        w = (cur_lane_width + next_lane_width) / 2.
        l = self.traj_parameters['lane_change_length']
        r = self.traj_parameters['lane_change_time_constant']
        tf = math.sqrt(l ** 2 + w ** 2) * r / v_cur

        # parameters for x
        dx = 0
        cx = v_cur
        ax = (2. * v_cur * tf - 2. * l) / (tf ** 3.)
        bx = -3. / 2 * ax * tf

        # parameters for y
        dy = 0
        cy = 0
        ay = -2. * w / (tf ** 3.)
        by = 3 * w / (tf ** 2.)

        # return result
        neutral_traj = []

        # time loop
        time_disc = self.traj_parameters['lane_change_time_disc']
        total_loop_count = int(tf / time_disc + 1)
        for i in range(total_loop_count):
            t = i * time_disc
            x_value = ax * (t ** 3.) + bx * (t ** 2.) + cx * t + dx
            y_value = -(ay * (t ** 3.) + by * (t ** 2.) + cy * t + dy)
            x_deriv = 3. * ax * (t ** 2.) + 2. * bx * t + cx
            y_deriv = -(3. * ay * (t ** 2.) + 2. * by * t + cy)
            theta = math.atan2(y_deriv, x_deriv)
            speed = math.sqrt(y_deriv ** 2. + x_deriv ** 2.)
            pose = PoseSpeedTemp()
            pose.speed = speed * 3.6
            pose.x = x_value
            pose.y = y_value
            pose.theta = theta
            neutral_traj.append(pose)

        return neutral_traj

    def laneChangeTraj(self, sim_data):
        cur_vehicle_pose = PoseTemp(sim_data.cur_vehicle_state.vehicle_location)

        # generate trajectory
        if not self.current_action == RLDecision.SWITCH_LANE:
            self.reset(RLDecision.SWITCH_LANE)
            # ToDo: Use closest pose for lane width
            neutral_traj = self.cubicSplineGen(sim_data.current_lane.lane[0].width, \
                            sim_data.next_lane.lane[0].width, sim_data.cur_vehicle_state.vehicle_speed)
            # determine the closest next pose in the current lane
            lane_pose_array = sim_data.current_lane.lane
            closest_pose = lane_pose_array[0].pose

            for lane_waypoint in lane_pose_array:
                closest_pose = lane_waypoint.pose
                way_pose = PoseTemp(lane_waypoint.pose)
                if way_pose.distance(cur_vehicle_pose) < TrajGenerator.SAME_POSE_THRESHOLD and \
                        way_pose.isInfrontOf(cur_vehicle_pose) and \
                        way_pose.distance(cur_vehicle_pose) > TrajGenerator.SAME_POSE_LOWER_THRESHOLD:
                    break
            closest_pose = PoseTemp(closest_pose)

            # offset the trajectory with the closest pose
            for pose_speed in neutral_traj:
                self.generated_path.append(pose_speed.addToPose(closest_pose))

            self.path_pointer = 0

        # find the next tracking point
        while (self.path_pointer < len(self.generated_path)):
            # traj pose
            pose_speed = self.generated_path[self.path_pointer]

            if pose_speed.isInfrontOf(cur_vehicle_pose) and \
                    pose_speed.distance(cur_vehicle_pose) > TrajGenerator.SAME_POSE_LOWER_THRESHOLD:
                break

            self.path_pointer += 1

        new_path_plan = PathPlan()
        # determine if lane switch is completed

        action_progress = self.path_pointer / float(len(self.generated_path))
        end_of_action = False
        #if rl_data.reset_run:
        #    end_of_action = True
        if action_progress >= 0.9999:
            end_of_action = True
            action_progress = 1.

        if end_of_action:
            self.reset()
        else:
            new_path_plan.tracking_pose.x = self.generated_path[self.path_pointer].x
            new_path_plan.tracking_pose.y = self.generated_path[self.path_pointer].y
            new_path_plan.tracking_pose.theta = self.generated_path[self.path_pointer].theta
            new_path_plan.tracking_speed = self.generated_path[self.path_pointer].speed
        new_path_plan.reset_sim = False
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = end_of_action

        # future poses
        path_pointer = self.path_pointer
        new_path_plan.future_poses = []
        while path_pointer < len(self.generated_path):
            new_pose_ros = Pose2D()
            new_pose_ros.x = self.generated_path[self.path_pointer].x
            new_pose_ros.y = self.generated_path[self.path_pointer].y
            new_pose_ros.theta = self.generated_path[self.path_pointer].theta
            new_path_plan.future_poses.append(new_pose_ros)
            path_pointer += 1

        return new_path_plan


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
