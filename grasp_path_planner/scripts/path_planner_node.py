#!/usr/bin/env python2.7

import rospy
import Queue
from enum import Enum
from ros_message_test.msg import LanePoint
from ros_message_test.msg import Lane
from ros_message_test.msg import VehicleState
from ros_message_test.msg import RewardInfo
from ros_message_test.msg import EnvironmentState
from ros_message_test.msg import RLCommand
from ros_message_test.msg import PathPlan


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

	def completePair(self):
		if self.rl_backlog.empty() or self.sim_backlog.empty():
			return False
		return True

	def newRLMessage(self, data):
		self.rl_backlog.put(RLDataProcessor(data))

	def newSimMessage(self, data):
		self.sim_backlog.put(SimDataProcessor(data))

	def newPair():
		rl_msg = rl_backlog.get()
		sim_msg = sim_backlog.get()
		return rl_msg, sim_msg

class TrajGenerator:
	def __init__(self, traj_parameters):
		self.traj_parameters = traj_parameters

backlog_manager = BacklogData()
TRAJ_PARAM = {'look_up_distance' : 10 ,\
		'lane_change_length' : 10,\
		'accelerate_amt' : 3,\
		'decelerate_amt' : 3,\
		'min_speed' : 0
}

class PathPlannerManager():
	def __init__(self):
		self.rl_sub = None
		self.sim_sub = None
		self.pub_path = None

	def rlCallback(self.data):
		backlog_manager.newRLMessage(data)
		self.pathPlanCallback()

	def simCallback(self.data):
		backlog_manager.newSimMessage(data)
		self.pathPlanCallback()

	def pathPlanCallback(self):
		# check if there is a new pair of messages
		if backlog_manager.completePair():
			# get the pair
			rl_data, sim_data = backlog_manager.newPair()

			# gernerate the path
		
			# publish the path

	def initialize(self):
		# initialize publisher
		self.pub_path = rospy.Publisher(PATH_PLAN_TOPIC_NAME, PathPlan, queue_size = 10)

		# initialize node
		rospy.init_node(NODE_NAME, anonymous=True)

		# refresh rate
		rate = rospy.Rate(500) # 500Hz
	
		# initialize subscriber
		self.rl_sub = rospy.Subscriber(RL_TOPIC_NAME, RLCommand, self.rlCallback)
		self.sim_sub = rospy.Subscriber(SIM_TOPIC_NAME, EnvironmentState, self.simCallback)

		rospy.spin()	
		
def pathPlannerMain():
	# initialize publisher
	pub_path = rospy.Publisher(PATH_PLAN_TOPIC_NAME, PathPlan, queue_size = 10)

	# initialize node
	rospy.init_node(NODE_NAME, anonymous=True)

	# refresh rate
	rate = rospy.Rate(500) # 500Hz
	
	# initialize subscriber
	rospy.Subscriber(RL_TOPIC_NAME, RLCommand, rlCallback)
	rospy.Subscriber(SIM_TOPIC_NAME, EnvironmentState, simCallback)

	rospy.spin()	
	while not rospy.is_shutdown():
		# check if there is a new pair of messages
		if backlog_manager.completePair():
			# get the data
			rl_data, sim_data = backlog_manager.newPair()
			
			# generate the path

			# publish the path
	
		# ros spin
		rospy.spin()

if __name__ == '__main__':
	try:
		pathPlannerMain()
	except rospy.ROSInterruptException:
		pass
