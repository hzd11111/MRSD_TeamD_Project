import os
import sys
import carla
import numpy as np

import rospy
import math

from geometry_msgs.msg import Pose2D as Pose2DMsg
from carla_utils.msg import ActorMsg, FrenetMsg

from functional_utility import Frenet

from frenet import Frenet #TODO: Fix Import
from lanestatus import LaneStatus # TODO:Fix import


class Actor():
    def __init__(self, world: carla.libcarla.World, actor_id: int):

        # system identifiers
        self.world = world
        self.actor_id = actor_id
        self.actor = None

        # TODO: delete if not needed
        # # variables to pass to the rosmsg
        # self.speed = None
        # self.acceleration = None
        # self.location_global = None
        # self.location_frenet = None
        # self.length = None
        # self.width = None
    
    def toROS(self):
        # Publishes the actor state info to ROS
        # int64 actor_id
        # float64 speed
        # float64 acceleration
        # geometry_msgs/Pose2D location_global
        # FrenetMsg location_frenet
        # float64 length
        # float64 width
        state_dict = self.get_state_dict()

        # create the Pose2D message for actor location
        pose_msg = Pose2DMsg()
        pose_msg.x = state_dict['Pose2D'][0]
        pose_msg.y = state_dict['Pose2D'][1]
        pose_msg.theta = state_dict['Pose2D'][2]

        # create the frenet message for actor location
        frenet_msg = FrenetMsg()
        frenet_msg.x = state_dict['location_frenet'][0]
        frenet_msg.y = state_dict['location_frenet'][1]
        frenet_msg.theta = state_dict['location_frenet'][2]

        # create the actor message
        msg = ActorMsg()
        msg.actor_id = state_dict['actor_id']
        msg.speed = state_dict['speed']
        msg.acceleration = state_dict['acceleration']
        msg.location_global = pose_msg
        msg.location_frenet = frenet_msg
        msg.length = state_dict['length']
        msg.width = state_dict['width']

        return msg

    def get_state_dict(self):
        '''
        Gets the actor status and returns a status dict with the following info:
        - actor_id (int)
        - speed (float)
        - acceleration (list [x,y,z]) 
        - global location (list [x,y,z])
        - location frenet (TODO)
        - length (float)
        - width (float)
        '''
        #TODO: how comprehensive do we want to be here?
        state_dict = {}
        state_dict['actor'] = self.actor
        state_dict['actor_id'] = self.actor_id

        # speed
        speed = self.actor.get_velocity()
        state_dict['speed_3d'] = [speed.x, speed.y, speed.z]
        state_dict['speed'] = np.linalg.norm([speed.x, speed.y, speed.z])

        # acceleration
        acc = self.actor.get_accleration()
        state_dict['acceleration_3d'] = [acc.x, acc.y, acc.z]
        state_dict['acceleration'] = np.linalg.norm([acc.x, acc.y, acc.z])

        # actor transform
        trans = actor.get_transform()

        # global location and rotation
        loc = trans.location
        rot = trans.rotation

        # frenet_x, frenet_y, theta = Frenet.get_frenet_distance_from_global(loc.x, loc.y, loc.z) #TODO
        frenet_x, frenet_y, theta = [0, 0, 0] #TODO: implement Frenet

        state_dict['global_location_3d'] = [loc.x, loc.y, loc.z]
        state_dict['global_location_2d'] = [loc.x, loc.y]
        state_dict['Pose2D'] = [loc.x, loc.y, rot.yaw]
        state_dict['location_frenet'] = [frenet_x, frenet_y, theta]

        # global rotation
        state_dict['rpy'] = [rot.roll, rot.pitch, rot.yaw]

        # actor dimensions
        state_dict['length'] = actor.bounding_box.extent.x
        state_dict['width'] = actor.bounding_box.extent.y
        state_dict['height'] = actor.bounding_box.extent.z

        return state_dict

    def spawn(self, location: Pose2D=None) -> carla.Actor:
        '''
        Constructor method to overwrite in the child class to spawn an instance of the child. 
        '''
        raise Exception("This needs to be implemented in the child class")

    def destroy(self):
        '''
        Method to destroy the actor.
        Returns:
            bool: True if successful
        '''
        return self.actor.destroy()

    def fromControllingVehicle(contolling_vehicle: Frenet, current_lane) -> Frenet:
        '''
        Gets the distance from the ego/controlling vehicle in frenet coordinate frame
        '''
        
        pass

    def getLocationFrenet():
        pass


class Vehicle(Actor):
    def __init__(self, world: carla.libcarla.World, actor_id: int):
        super.__init__(world, actor_id)
        pass

    @classmethod
    def fromRosMsg(cls, msg):
        raise NotImplemented


class Pedestrian(Actor):
    def __init__(self, world: carla.libcarla.World, actor_id: int):
        super.__init__(world, actor_id)
        pass

    @classmethod
    def fromRosMsg(cls, msg):
        raise NotImplemented