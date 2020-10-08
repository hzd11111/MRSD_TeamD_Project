#!/usr/bin/env python
import os
import sys
import numpy as np
import rospy
import math

from functional_utility import Frenet, Pose2D
from geometry_msgs.msg import Pose2D as Pose2DMsg
from carla_utils.msg import ActorMsg, VehicleMsg, PedestrainMsg, FrenetMsg
from options import TrafficLightStatus, PedestrainPriority


class Actor:
    def __init__(
        self,
        world=None,
        actor_id=0,
        speed=0.0,
        acceleration=0.0,
        location_global=Pose2D(),
        location_frenet=Frenet(),
        length=0.0,
        width=0.0,
    ):

        # system identifiers
        self.world = world
        self.actor_id = actor_id
        self.actor = None

        # variables to pass to the rosmsg
        self.speed = speed
        self.acceleration = acceleration
        self.location_global = location_global
        self.location_frenet = location_frenet
        self.length = length
        self.width = width

    @classmethod
    def fromRosMsg(cls, actor_msg):
        # obj = cls.__new__(cls)
        obj = cls()
        obj.speed = actor_msg.speed
        obj.acceleration = actor_msg.acceleration
        obj.location_global = Pose2D.fromRosMsg(actor_msg.location_global)
        obj.location_frenet = Frenet.fromRosMsg(actor_msg.location_frenet)
        obj.length = actor_msg.length
        obj.width = actor_msg.width
        return obj

    def toRosMsg(self):
        """
        Returns the Actor ROS msg. If world is available, updates the state variable before returning.
        """
        msg = ActorMsg()
        if self.world is not None:
            self.actor = self.world.get_actor(self.actor_id)

            # Publishes the actor state info to ROS
            state_dict = self.get_state_dict()

            # create the Pose2D message for actor location
            pose_msg = Pose2DMsg()
            pose_msg.x = state_dict["Pose2D"][0]
            pose_msg.y = state_dict["Pose2D"][1]
            pose_msg.theta = state_dict["Pose2D"][2]

            # create the frenet message for actor location
            frenet_msg = FrenetMsg()
            frenet_msg.x = state_dict["location_frenet"][0]
            frenet_msg.y = state_dict["location_frenet"][1]
            frenet_msg.theta = state_dict["location_frenet"][2]

            # create the actor message
            msg.actor_id = state_dict["actor_id"]
            msg.speed = state_dict["speed"]
            msg.acceleration = state_dict["acceleration"]
            # msg.location_global = pose_msg
            # msg.location_frenet = frenet_msg
            msg.length = state_dict["length"]
            msg.width = state_dict["width"]
        else:
            msg.actor_id = self.actor_id
            msg.speed = self.speed
            msg.acceleration = self.acceleration
            msg.length = self.length
            msg.width = self.width

        msg.location_global = self.location_global.toRosMsg()
        msg.location_frenet = self.location_frenet.toRosMsg()

        return msg

    def get_state_dict(self, actor=None):
        """
        Gets the actor status and returns a status dict with the following info:
        - actor_id (int)
        - speed (float)
        - acceleration (list [x,y,z])
        - global location (list [x,y,z])
        - location frenet (TODO)
        - length (float)
        - width (float)
        """
        if actor is None:
            actor = self.actor

        state_dict = {}
        state_dict["actor_id"] = actor.id
        state_dict["type_id"] = actor.type_id

        # speed
        speed = actor.get_velocity()
        state_dict["speed_3d"] = [speed.x, speed.y, speed.z]
        state_dict["speed"] = np.linalg.norm([speed.x, speed.y, speed.z])

        # acceleration
        acc = actor.get_acceleration()
        state_dict["acceleration_3d"] = [acc.x, acc.y, acc.z]
        state_dict["acceleration"] = np.linalg.norm([acc.x, acc.y, acc.z])

        # actor transform
        trans = actor.get_transform()

        # global location and rotation
        loc = trans.location
        rot = trans.rotation

        # frenet_x, frenet_y, theta = Frenet.get_frenet_distance_from_global(loc.x, loc.y, loc.z) #TODO
        frenet_x, frenet_y, theta = [0, 0, 0]  # TODO: implement Frenet

        state_dict["global_location_3d"] = [loc.x, loc.y, loc.z]
        state_dict["global_location_2d"] = [loc.x, loc.y]
        state_dict["Pose2D"] = [loc.x, loc.y, rot.yaw]
        state_dict["location_frenet"] = [frenet_x, frenet_y, theta]

        # global rotation
        state_dict["rpy"] = [rot.roll, rot.pitch, rot.yaw]

        # actor dimensions
        state_dict["length"] = actor.bounding_box.extent.x * 2
        state_dict["width"] = actor.bounding_box.extent.y * 2
        state_dict["height"] = actor.bounding_box.extent.z * 2

        return state_dict

    def spawn(self, Pose2D=None):
        """
        Constructor method to overwrite in the child class to spawn an instance of the child.
        """
        raise Exception("This needs to be implemented in the child class")

    def destroy(self):
        """
        Method to destroy the actor.
        Returns:
            bool: True if successful
        """
        return self.actor.destroy()

    def fromControllingVehicle(self, Frenet, current_lane):
        """
        Gets the distance from the ego/controlling vehicle in frenet coordinate frame
        """
        pass

    def getLocationFrenet(self):
        pass


class Vehicle(Actor):
    def __init__(
        self,
        world=None,
        actor_id=0,
        speed=0.0,
        acceleration=0.0,
        location_global=Pose2D(),
        location_frenet=Frenet(),
        length=0.0,
        width=0.0,
        traffic_light_status=None,
    ):

        super(Vehicle, self).__init__(
            world,
            actor_id,
            speed,
            acceleration,
            location_global,
            location_frenet,
            length,
            width,
        )
        self.traffic_light_status = traffic_light_status
        if self.traffic_light_status is None:
            self.traffic_light_status = TrafficLightStatus.RED

    @classmethod
    def fromRosMsg(cls, msg):
        obj = cls.__new__(cls)
        obj = super(Vehicle, obj).fromRosMsg(msg.actor_msg)
        obj.traffic_light_status = TrafficLightStatus(msg.traffic_light_status)
        return obj

    def toRosMsg(self):
        msg = VehicleMsg()
        msg.actor_msg = super(Vehicle, self).toRosMsg()
        msg.traffic_light_status = self.traffic_light_status.value
        return msg


class Pedestrian(Actor):
    def __init__(
        self,
        world=None,
        actor_id=0,
        speed=0.0,
        acceleration=0.0,
        location_global=Pose2D(),
        location_frenet=Frenet(),
        length=0.0,
        width=0.0,
        priority_status=None
    ):
        super(Pedestrian, self).__init__(
            world,
            actor_id,
            speed,
            acceleration,
            location_global,
            location_frenet,
            length,
            width,
        )
        self.priority_status = priority_status
        if self.priority_status is None:
            self.priority_status = PedestrainPriority.JAYWALK

    @classmethod
    def fromRosMsg(cls, msg):
        obj = cls.__new__(cls)
        obj = super(Pedestrian, obj).fromRosMsg(msg.actor_msg)
        obj.priority_status = PedestrainPriority(msg.priority_status)
        return obj

    def toRosMsg(self):
        msg = PedestrainMsg()
        msg.actor_msg = super(Pedestrian, self).toRosMsg()
        msg.priority_status = self.priority_status.value
        return msg


TEST_NODE_NAME = "actor_listener"
TEST_TOPIC_NAME = "custom_chatter"

# Test Code for Deserialize and Infor Printing
def callback(data):
    # path_plan = PathPlan.fromRosMsg(data)
    # print("receiving data")
    # rospy.loginfo("%f is age: %d" % (data.tracking_speed, data.reset_sim))
    obj = Pedestrian.fromRosMsg(data)
    print("Confirm msg is: ", obj.location_frenet.theta, obj.location_global.x, obj.location_global.y, obj.location_global.theta)


def listener():
    rospy.init_node(TEST_NODE_NAME, anonymous=True)
    rospy.Subscriber(TEST_TOPIC_NAME, PedestrainMsg, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == "__main__":
    listener()