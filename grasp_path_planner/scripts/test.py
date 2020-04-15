# import sys
# import os
# import time
# homedir=os.getenv("HOME")
# distro=os.getenv("ROS_DISTRO")
# sys.path.remove("/opt/ros/"+distro+"/lib/python2.7/dist-packages")
# sys.path.append("/opt/ros/"+distro+"/lib/python2.7/dist-packages")
# # to remove tensorflow warnings
import rosbag
from full_planner_node import Reward
# from msg.VehicleState import VehicleState

messages = rosbag.Bag("msg.bag")
env_msgs = [msg for _,msg,_ in messages.read_messages(['/environment_state'])]
data = env_msgs[100]
reward_manager = Reward()
reward_manager.update(data,1)