#!/usr/bin/env python
import math
import rospy

from geometry_msgs.msg import Pose2D as Pose2DMsg
from carla_utils.msg import FrenetMsg


class Frenet(object):
    __slots__ = ["x", "y", "theta"]

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta

    @classmethod
    def fromRosMsg(cls, msg):
        cls.x = msg.x
        cls.y = msg.y
        cls.theta = msg.theta
        return cls

    def toRosMsg(self):
        msg = FrenetMsg()
        msg.x = self.x
        msg.y = self.y
        msg.theta = self.theta
        return msg

    def perpendicularFromCurrent(self):
        pass

    def parallelFromCurrent(self):
        pass


class Vec2D:
    __slots__ = ["x", "y"]

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def add(self, other):
        return Vec2D(self.x + other.x, self.y + other.y)

    def sub(self, other):
        return Vec2D(self.x - other.x, self.y - other.y)

    def dot(self, other):
        upper = self.x * other.x + self.y * other.y
        lower = self.norm() * other.norm()
        if lower <= 0.00001:
            return 1
        return upper / lower


class Pose2D(object):
    __slots__ = ["x", "y", "theta"]

    def __init__(self, x: float = 0, y: float = 0, theta: float = 0):
        self.x = x
        self.y = y
        self.theta = theta

    @classmethod
    def fromRosMsg(cls, msg):
        cls.x = msg.x
        cls.y = msg.y
        cls.theta = msg.theta
        return cls

    def toRosMsg(self):
        msg = Pose2DMsg()
        msg.x = self.x
        msg.y = self.y
        msg.theta = self.theta
        return msg

    def wrapToPi(self, theta: float) -> float:
        return (theta + math.pi) % (2.0 * math.pi) - math.pi

    def distance(self, pose) -> float:
        return math.sqrt((self.x - pose.x) ** 2.0 + (self.y - pose.y) ** 2.0)

    def add(self, pose):
        new_pose = Pose2D()
        new_pose.x = (
            self.x + pose.x * math.cos(self.theta) - pose.y * math.sin(self.theta)
        )
        new_pose.y = (
            self.y + pose.x * math.sin(self.theta) + pose.y * math.cos(self.theta)
        )
        new_pose.theta = self.wrapToPi(self.theta + pose.theta)
        return new_pose

    def vecTo(self, pose) -> Vec2D:
        new_vec = Vec2D()
        new_vec.x = pose.x - self.x
        new_vec.y = pose.y - self.y
        return new_vec

    def vecFromTheta(self) -> Vec2D:
        return Vec2D(math.cos(self.theta), math.sin(self.theta))

    def isInfrontOf(self, pose) -> bool:
        diff_vec = pose.vecTo(self)
        other_vec = pose.vecFromTheta()
        return diff_vec.dot(other_vec) > 0

    def scalarMultiply(self, scalar: float):
        new_pose = Pose2D()
        new_pose.x = self.x * scalar
        new_pose.y = self.y * scalar
        new_pose.theta = self.theta * scalar


class PoseSpeed(Pose2D):
    __slots__ = ["speed"]

    def __init__(self, speed: float = 0):
        Pose2D.__init__(self)
        self.speed = speed

    def addToPose(self, pose: Pose2D):
        new_pose = PoseSpeed()
        new_pose.x = (
            pose.x + self.x * math.cos(pose.theta) - self.y * math.sin(pose.theta)
        )
        new_pose.y = (
            pose.y + self.x * math.sin(pose.theta) + self.y * math.cos(pose.theta)
        )
        new_pose.theta = self.wrapToPi(self.theta + pose.theta)
        new_pose.speed = self.speed
        return new_pose
