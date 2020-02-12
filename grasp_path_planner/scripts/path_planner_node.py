#!/usr/bin/env python2.7

import rospy
import Queue
import math
import copy

from enum import Enum
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import RLCommand
from grasp_path_planner.msg import PathPlan


NODE_NAME = 'path_planner'
SIM_TOPIC_NAME = "environment_state"
RL_TOPIC_NAME = "rl_decision"
PATH_PLAN_TOPIC_NAME = "path_plan"
class RLDecision(Enum):
	CONSTANT_SPEED = 1
	ACCELERATE = 2
	DECELERATE = 3
	SWITCH_LANE = 4

class RLDataProcessor:
	def __init__(self, rl_data):
		const_speed = rl_data.constant_speed
		acc = rl_data.accelerate
		dec = rl_data.decelerate
		cl = rl_data.change_lane
		self.reset_run = rl_data.reset_run
		self.id = rl_data.id
		if not ((const_speed + acc + dec + cl) == 1):
			print("RL Message Content Error")
		if const_speed:
			self.rl_decision = RLDecision.CONSTANT_SPEED
		elif acc:
			self.rl_decision = RLDecision.ACCELERATE
		elif dec:
			self.rl_decision = RLDecision.DECELERATE
		else:
			self.rl_decision = RLDecision.SWITCH_LANE

	@property
	def rl_decision(self):
		return self.__rl_decision

	@property
	def reset_run(self):
		return self.__reset_run

	@rl_decision.setter
	def rl_decision(self, rl_decision):
		self.__rl_decision = rl_decision	

	@reset_run.setter
	def reset_run(self, reset_run):
		self.__reset_run = reset_run

class SimDataProcessor:
	def __init__(self, sim_data):
		self.cur_lane = sim_data.current_lane
		self.next_lane = sim_data.next_lane
		self.ego_vehicle = sim_data.cur_vehicle_state
		self.id = sim_data.id
	@property
	def cur_lane(self):
		return self.__cur_lane

	@property
	def next_lane(self):
		return self.__next_lane

	@property
	def ego_vehicle(self):
		return self.__ego_vehicle

	@cur_lane.setter
	def cur_lane(self, cur_lane):
		self.__cur_lane = cur_lane

	@next_lane.setter
	def next_lane(self, next_lane):
		self.__next_lane = next_lane
	
	@ego_vehicle.setter
	def ego_vehicle(self, ego_vehicle):
		self.ego_vehicle = ego_vehicle

class BacklogData:
	def __init__(self):
		self.rl_backlog = Queue.Queue()
		self.sim_backlog = Queue.Queue()
	def reset(self):
		with self.rl_backlog.mutex:
			self.rl_backlog.clear()
		with self.sim_backlog.mutex:
			self.sim_backlog.clear()		

	def completePair(self):
		if self.rl_backlog.empty() or self.sim_backlog.empty():
			return False
		return True

	def newRLMessage(self, data):
		self.rl_backlog.put(RLDataProcessor(data))

	def newSimMessage(self, data):
		self.sim_backlog.put(SimDataProcessor(data))

	def newPair(self):
		rl_msg = self.rl_backlog.get()
		sim_msg = self.sim_backlog.get()
		return rl_msg, sim_msg

class VecTemp:
	def __init__(self, x=0, y=0):
		self.x = x
		self.y = y

	#def __init__(self, x, y):
	#	self.x = x
	#	self.y = y

	#def __init__(self, pose):
	#	self.x = pose.x
	#	self.y = pose.y

	def norm(self):
		return math.sqrt(self.x **2 + self.y**2)

	def add(self, other):
		return VecTemp(self.x+other.x, self.y+other.y)
	
	def sub(self, other):
		return VecTemp(self.x-other.x, self.y-other.y)

	def dot(self, other):
		upper = self.x*other.x + self.y+other.y
		lower = self.norm() * other.norm()
		if lower <= 0.00001:
			return 1
		return upper / lower
		
	
class PoseTemp:
	def __init__(self, ros_pose=False):
		if ros_pose:
			self.x=ros_pose.x
			self.y=ros_pose.y
			self.theta=self.wrapToPi(ros_pose.theta)
		else:
			self.x = 0
			self.y = 0
			self.theta = 0

	#def __init__(self, ros_pose):
	#	self.x = ros_pose.x
	#	self.y = ros_pose.y
	#	self.theta = self.wrapToPi(ros_pose.theta)

	def wrapToPi(self, theta):
		return (theta + math.pi) % (2. * math.pi) - math.pi

	def distance(self, pose):
		return math.sqrt((self.x - pose.x) **2. + (self.y - pose.y) ** 2.)

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
	#def __init__(self, pose, speed):
	#	self.speed = speed
	#	PoseTemp.__init__(self,pose)
	def addToPose(self, pose):
		new_pose = PoseSpeedTemp()
		new_pose.x = self.x + pose.x * math.cos(self.theta) - pose.y * math.sin(self.theta)
		new_pose.y = self.y + pose.x * math.sin(self.theta) + pose.y * math.cos(self.theta)
		new_pose.theta = self.wrapToPi(self.theta + pose.theta)
		new_pose.speed = self.speed
		return new_pose

class TrajGenerator:
	SAME_POSE_THRESHOLD = 1
	def __init__(self, traj_parameters):
		self.traj_parameters = traj_parameters
		self.reset()
	
	def reset(self):
		self.lane_switching = False
		self.generated_path = []
		self.path_pointer = 0

	def trajPlan(self, rl_data, sim_data):
		if len(sim_data.cur_lane.lane) <= 0 or len(sim_data.next_lane.lane) <= 0:
			print("Lane has no component")

		if self.lane_switching:
			return self.laneChangeTraj(rl_data, sim_data)
		elif rl_data.rl_decision == RLDecision.CONSTANT_SPEED:
			return self.constSpeedTraj(rl_data, sim_data)
		elif rl_data.rl_decision == RLDecision.ACCELERATE:
			return self.constSpeedTraj(rl_data, sim_data)
		elif rl_data.rl_decision == RLDecision.DECELERATE:
			return self.constSpeedTraj(rl_data, sim_data)
		elif rl_data.rl_decision == RLDecision.SWITCH_LANE:
			return self.laneChangeTraj(rl_data, sim_data)
		else:
			print("RL Decision Error")

	def constSpeedTraj(self, rl_data, sim_data):
		
		# current vehicle state
		cur_vehicle_state = sim_data.ego_vehicle
		cur_vehicle_pose = PoseTemp(cur_vehicle_state.vehicle_location)
		cur_vehicle_speed = cur_vehicle_state.vehicle_speed
	
		# determine the closest next pose in the current lane
		lane_pose_array = sim_data.cur_lane.lane
		closest_pose = lane_pose_array[0]

		for lane_waypoint in lane_pose_array:
			closest_pose = lane_waypoint
			way_pose = PoseTemp(lane_waypoint.pose)
			if way_pose.distance(cur_vehicle_pose) < TrajGenerator.SAME_POSE_THRESHOLD and\
				way_pose.isInfrontOf(cur_vehicle_pose):
				break	

		new_path_plan = PathPlan()
		new_path_plan.tracking_pose = closest_pose.pose
		new_path_plan.reset_sim = rl_data.reset_run
		#new_path_plan.tracking_speed = cur_vehicle_speed
		new_path_plan.tracking_speed = 20
		return new_path_plan

	def cubicSplineGen(self, cur_lane_width, next_lane_width, v_cur):
		if v_cur < 10:
			v_cur = 10
		# determine external parameters
		w = (cur_lane_width + next_lane_width)/2.
		l = self.traj_parameters['lane_change_length']
		r = self.traj_parameters['lane_change_time_constant']
		tf = math.sqrt(l**2 + w**2) * r / v_cur

		# parameters for x
		dx = 0
		cx = v_cur
		ax = (2.*v_cur * tf - 2.*l) / (tf ** 3.)
		bx = -3./2 * ax * tf

		# parameters for y
		dy = 0
		cy = 0
		ay = -2.*w/(tf **3.)
		by = 3*w/(tf **2.)

		# return result
		neutral_traj = []

		# time loop
		time_disc = self.traj_parameters['lane_change_time_disc'] 
		total_loop_count = int(tf / time_disc + 1)
		for i in range(total_loop_count):
			t = i * time_disc
			x_value = ax*(t**3.)+bx*(t**2.)+cx*t+dx
			y_value = ay*(t**3.)+by*(t**2.)+cy*t+dy
			x_deriv = 3.*ax*(t**2.)+2.*bx*t+cx
			y_deriv = 3.*ay*(t**2.)+2.*by*t+cy
			theta = math.atan2(y_deriv,x_deriv)
			speed = math.sqrt(y_deriv**.2+x_deriv**.2)
			pose = PoseSpeedTemp()
			pose.speed = speed
			pose.x = x_value
			pose.y = y_value
			pose.theta = theta
			neutral_traj.append(pose)
		
		return neutral_traj
		

	def laneChangeTraj(self, rl_data, sim_data):
		cur_vehicle_pose = PoseTemp(sim_data.ego_vehicle.vehicle_location)
		
		# generate trajectory
		if not self.lane_switching:
			# ToDo: Use closest pose for lane width
			neutral_traj = self.cubicSplineGen(sim_data.cur_lane.lane[0].width,\
						sim_data.next_lane.lane[0].width, sim_data.ego_vehicle.vehicle_speed)

			# determine the closest next pose in the current lane
			lane_pose_array = sim_data.cur_lane.lane
			closest_pose = lane_pose_array[0].pose

			for lane_waypoint in lane_pose_array:
				closest_pose = lane_waypoint.pose
				way_pose = PoseTemp(lane_waypoint.pose)
				if way_pose.distance(cur_vehicle_pose) < TrajGenerator.SAME_POSE_THRESHOLD and\
					way_pose.isInfrontOf(cur_vehicle_pose):
					break	
			closest_pose = PoseTemp(closest_pose)

			# offset the trajectory with the closest pose
			for pose_speed in neutral_traj:
				self.generated_path.append(pose_speed.addToPose(closest_pose))
	
			# change lane switching status	
			self.lane_switching = True

		# find the next tracking point
		while (self.path_pointer < len(self.generated_path)):
			# traj pose
			pose_speed = self.generated_path[self.path_pointer]

			if pose_speed.isInfrontOf(cur_vehicle_pose):
				break

			self.path_pointer += 1
		
			
		new_path_plan = PathPlan()
		# determine if lane switch is completed
		if self.path_pointer >= len(self.generated_path):
			# reset the trajectory
			self.reset()
			new_path_plan.reset_sim = 1
			return new_path_plan	
		
		new_path_plan.tracking_pose.x = self.generated_path[self.path_pointer].x
		new_path_plan.tracking_pose.y = self.generated_path[self.path_pointer].y
		new_path_plan.tracking_pose.theta = self.generated_path[self.path_pointer].theta
		new_path_plan.reset_sim = 0
		new_path_plan.tracking_speed = self.generated_path[self.path_pointer].speed
		return new_path_plan		

TRAJ_PARAM = {'look_up_distance' : 0 ,\
		'lane_change_length' : 30,\
		'lane_change_time_constant' : 1.05,\
		'lane_change_time_disc' : 0.05,\
		'accelerate_amt' : 3,\
		'decelerate_amt' : 3,\
		'min_speed' : 0
}

class PathPlannerManager:
	def __init__(self):
		self.rl_sub = None
		self.sim_sub = None
		self.pub_path = None
		self.traj_gen = None
		self.backlog_manager = BacklogData()
		self.calls = 0

	def rlCallback(self, data):
		print("RL ID Received:",data.id)
		self.backlog_manager.newRLMessage(copy.copy(data))
		print("RL ID Added:",data.id)
		self.pathPlanCallback()

	def simCallback(self, data):
		print("Simulation ID Received:",data.id)
		self.backlog_manager.newSimMessage(copy.copy(data))
		print("Simulation ID Added:",data.id)
		self.pathPlanCallback()

	def pathPlanCallback(self):
		# check if there is a new pair of messages
		if self.backlog_manager.completePair():

			# get the pair
			rl_data, sim_data = self.backlog_manager.newPair()
			print"Publishing for sim:", sim_data.id, "rl:", rl_data.id
			# gernerate the path
			traj = self.traj_gen.trajPlan(rl_data, sim_data)			
		
			# reset the backlog if simulation needs to be reset
			if traj.reset_sim:
				self.backlog_manager.reset()
		
			# publish the path
			self.pub_path.publish(traj)

			

	def initialize(self):
		# initialize publisher
		self.pub_path = rospy.Publisher(PATH_PLAN_TOPIC_NAME, PathPlan, queue_size = 10)

		# initialize node
		rospy.init_node(NODE_NAME, anonymous=True)

		# initialize subscriber
		self.rl_sub = rospy.Subscriber(RL_TOPIC_NAME, RLCommand, self.rlCallback)
		self.sim_sub = rospy.Subscriber(SIM_TOPIC_NAME, EnvironmentState, self.simCallback)

		# initialize trajectory generator
		self.traj_gen = TrajGenerator(TRAJ_PARAM)

		# spin
		rospy.spin()	
		
if __name__ == '__main__':
	try:
		path_planner_main = PathPlannerManager()
		path_planner_main.initialize()
	except rospy.ROSInterruptException:
		pass
