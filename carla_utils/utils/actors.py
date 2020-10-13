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
        self.actor = None if self.world is None else world.get_actor(self.actor_id)

        # variables to pass to the rosmsg
        self.speed = speed if self.world is None else self.get_velocity()
        self.acceleration = (
            acceleration if self.world is None else self.get_acceleration()
        )
        self.location_global = (
            location_global if self.world is None else self.get_location_global()
        )
        self.location_frenet = (
            location_frenet if self.world is None else self.get_location_frenet()
        )
        self.length = length if self.world is None else self.get_length()
        self.width = width if self.world is None else self.get_width()

        # userful class attributes
        self.location = None if self.world is None else self.get_location()

    # -------ROS HELPER METHODS----------------------

    @classmethod
    def fromRosMsg(cls, actor_msg):
        obj = cls.__new__(cls)
        obj.__init__()
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
            pose_msg.x = state_dict["Pose2D"].x
            pose_msg.y = state_dict["Pose2D"].y
            pose_msg.theta = state_dict["Pose2D"].theta

            # create the frenet message for actor location
            # frenet_msg = FrenetMsg()
            # frenet_msg.x = state_dict["location_frenet"].x
            # frenet_msg.y = state_dict["location_frenet"].y
            # frenet_msg.theta = state_dict["location_frenet"].theta
            frenet_msg = self.location_frenet.toRosMsg()

            # create the actor message
            msg.actor_id = state_dict["actor_id"]
            msg.speed = state_dict["speed"]
            msg.acceleration = state_dict["acceleration"]
            msg.location_global = pose_msg
            msg.location_frenet = frenet_msg
            msg.length = state_dict["length"]
            msg.width = state_dict["width"]
        else:
            msg.actor_id = self.actor_id
            msg.speed = self.speed
            msg.acceleration = self.acceleration
            msg.location_global = self.location_global.toRosMsg()
            msg.location_frenet = self.location_frenet.toRosMsg()
            msg.length = self.length
            msg.width = self.width

        return msg

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

    def fromControllingVehicle(self, frenet, current_lane):
        """
        Gets the distance from the ego/controlling vehicle in frenet coordinate frame
        """
        if self.actor is None:
            current_global_pose = self.location_global
        else:
            current_global_pose = self.get_state_dict()["Pose2D"]

        current_frenet_pose = current_lane.GlobalToFrenet(current_global_pose)

        relative_s = current_frenet_pose.x - frenet.x
        # relative_d = current_frenet_pose.y - frenet.y
        # relative_theta = current_frenet_pose.theta - frenet.theta
        return Frenet(
            x=relative_s, y=current_frenet_pose.y, theta=current_frenet_pose.theta
        )

    # ---------GETTERS------------------------------

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
        state_dict["speed"] = np.linalg.norm([speed.x, speed.y, speed.z]) * 3.6

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

        state_dict["location_3d"] = {"x": loc.x, "y": loc.y, "z": loc.z}
        state_dict["location_2d"] = {"x": loc.x, "y": loc.y}
        state_dict["Pose2D"] = Pose2D(loc.x, loc.y, rot.yaw * np.pi / 180)
        state_dict["location_frenet"] = Frenet(frenet_x, frenet_y, theta)

        # global rotation
        state_dict["rpy"] = [rot.roll, rot.pitch, rot.yaw * np.pi / 180]

        # actor dimensions
        state_dict["length"] = actor.bounding_box.extent.x * 2
        state_dict["width"] = actor.bounding_box.extent.y * 2
        state_dict["height"] = actor.bounding_box.extent.z * 2

        return state_dict

    def getLocationFrenet(self):
        pass

    def get_velocity(self):
        return self.get_state_dict()["speed"]

    def get_acceleration(self):
        return self.get_state_dict()["acceleration"]

    def get_location_global(self):
        """Returns a Pose2D object"""
        return self.get_state_dict()["Pose2D"]

    def get_location_frenet(self):
        """Returns a Frenet object"""
        return self.get_state_dict()["location_frenet"]

    def get_length(self):
        return self.get_state_dict()["length"]

    def get_width(self):
        return self.get_state_dict()["width"]

    def get_location(self):
        return self.get_state_dict()["location_3d"]


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

    @staticmethod
    def getClosest(adjacent_lane_vehicles, ego_vehicle, n=5):
        """
        Gets the n closest vehicles to the ego vehicle.
        Input:
            adjacent_lane_vehicles(list): list of vehicle objects of class Vehicle
            ego_vehicle (Vehicle): ego vehicle of class Vehicle
            n(int): closest n objects
        Output:
            closest_n_vehicles (list): list of n closest vehicles
            sorted_idx (list): list of n vehicles indexes sorted by increasing distance from ego
        """
        ego_x = ego_vehicle.location["x"]
        ego_y = ego_vehicle.location["y"]

        distances = [
            (
                (ego_x - adjacent_lane_vehicles[i].location["x"]) ** 2
                + (ego_y - adjacent_lane_vehicles[i].location["y"]) ** 2
            )
            for i in range(len(adjacent_lane_vehicles))
        ]
        sorted_idx = np.argsort(distances)[:n]
        closest_n_vehicles = [adjacent_lane_vehicles[i] for i in sorted_idx]

        return closest_n_vehicles, sorted_idx


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
        priority_status=None,
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

    @staticmethod
    def getClosestPedestrian(pedestrians, ego_vehicle, n=1):
        # TODO: ROHAN move to the Pedestrian class
        ego_x = ego_vehicle.location["x"]
        ego_y = ego_vehicle.location["y"]

        distances = [
            (
                (ego_x - pedestrians[i].location["x"]) ** 2
                + (ego_y - pedestrians[i].location["y"]) ** 2
            )
            for i in range(len(pedestrians))
        ]
        sorted_idx = np.argsort(distances)[:n]
        closest_n_pedestrians = [pedestrians[i] for i in sorted_idx]

        return closest_n_pedestrians, sorted_idx


TEST_NODE_NAME = "actor_listener"
TEST_TOPIC_NAME = "custom_chatter"

# Test Code for Deserialize and Infor Printing
def callback(data):
    # path_plan = PathPlan.fromRosMsg(data)
    # print("receiving data")
    # rospy.loginfo("%f is age: %d" % (data.tracking_speed, data.reset_sim))
    obj = Pedestrian.fromRosMsg(data)
    print(
        "Confirm msg is: ",
        obj.location_frenet.theta,
        obj.location_global.x,
        obj.location_global.y,
        obj.location_global.theta,
    )


def listener():
    rospy.init_node(TEST_NODE_NAME, anonymous=True)
    rospy.Subscriber(TEST_TOPIC_NAME, PedestrainMsg, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == "__main__":
    listener()
