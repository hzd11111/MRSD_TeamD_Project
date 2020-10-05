#!/usr/bin/env python
import rospy
from carla_utils.msg import FrenetMsg


class Frenet(object):
    __slots__ = ["x", "y", "theta"]

    def __init__(self, x, y, theta):
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

    def perpendicularFromCurrent():
        pass

    def parallelFromCurrent():
        pass