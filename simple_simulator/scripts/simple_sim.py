#!/usr/bin/env python
import numpy as np
import rospy
import random
import threading

from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import PathPlan

from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker

NODE_NAME = 'simple_simulator'
SIM_TOPIC_NAME = 'environment_state'
PATH_PLAN_TOPIC_NAME = 'path_plan'
LANE_MARKER_TOPIC_NAME = 'lane_marker'
EGO_MARKER_TOPIC_NAME = 'ego_vehicle_marker'
VEHICLE_MARKER_TOPIC_NAME = "vehicle_markers"


class LaneSim:
    def __init__(self, starting_x, starting_y, starting_theta, length, width):
        self.starting_x = starting_x
        self.starting_y = starting_y
        self.starting_theta = starting_theta
        self.length = length
        self.width = width


class Vehicle:
    def __init__(self, length, width):
        self.length = length
        self.width = width
        self.x = 0
        self.y = 0
        self.theta = 0
        self.speed = 0

    def place(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def step(self, duration):
        self.x += self.speed * np.cos(self.theta) * duration
        self.y -= self.speed * np.sin(self.theta) * duration

    def setSpeed(self, speed):
        self.speed = speed


class SimpleSimulator:
    def __init__(self, time_step):
        self.vehicles = []
        self.controlling_vehicle = None
        self.lanes = []
        self.cur_lane = 0
        self.time_step = time_step

        self.env_msg = None
        self.lane_marker = None
        self.ego_marker = None
        self.vehicle_marker = None

        self.lock = threading.Lock()
        self.env_pub = None
        self.lane_pub = None
        self.ego_pub = None
        self.vehicle_pub = None
        self.path_sub = None

        # id
        self.id = 0
        self.id_waiting = 0

    def initialize(self):
        # initialize node
        rospy.init_node(NODE_NAME, anonymous=True)

        # initialize publisher
        self.env_pub = rospy.Publisher(SIM_TOPIC_NAME, EnvironmentState, queue_size=1000)

        self.lane_pub = rospy.Publisher(LANE_MARKER_TOPIC_NAME, MarkerArray, queue_size=1000)
        self.ego_pub = rospy.Publisher(EGO_MARKER_TOPIC_NAME, Marker, queue_size=1000)
        self.vehicle_pub = rospy.Publisher(VEHICLE_MARKER_TOPIC_NAME, MarkerArray, queue_size=1000)

        # initialize subscriber
        self.path_sub = rospy.Subscriber(PATH_PLAN_TOPIC_NAME, PathPlan, self.pathCallback)

        # reset scene
        self.resetScene()

        # send the first message
        self.updateMarkers()
        self.publishMessages()

    def pathCallback(self, msg):
        pass

    def spin(self):
        print("Start Ros Spin")
        # spin
        rospy.spin()

    def publishMessages(self):

        self.lock.acquire()

        if self.env_msg:
            self.env_pub.publish(self.env_msg)
        if self.lane_marker:
            self.lane_pub.publish(self.lane_marker)
        if self.ego_marker:
            self.ego_pub.publish(self.ego_marker)
        if self.vehicle_marker:
            self.vehicle_pub.publish(self.vehicle_marker)

        self.lock.release()

    def updateMessages(self):
        self.lock.acquire()

        # lanes

        # next lane vehicles

        # front vehicle

        # back vehicle

        # ego vehicle

        # reward info
        self.lock.release()

    def updateMarkers(self):
        self.lock.acquire()
        marker_id = 0
        # vehicles
        self.vehicle_marker = MarkerArray()
        for v in self.vehicles:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = marker.CUBE
            marker.action = marker.ADD
            marker.scale.x = v.length
            marker.scale.y = v.width
            marker.scale.z = 1.5
            marker.pose.orientation.w = 1.0 # ToDo: add in orientation
            marker.pose.position.x = v.x
            marker.pose.position.y = -v.y
            marker.pose.position.z = 0.75
            marker.color.g = 1.0
            marker.color.a = 1.0
            marker.id = marker_id
            marker_id += 1
            self.vehicle_marker.markers.append(marker)

        # ego vehicle
        self.ego_marker = Marker()
        self.ego_marker.header.frame_id = "map"
        self.ego_marker.type = self.ego_marker.CUBE
        self.ego_marker.action = self.ego_marker.ADD
        self.ego_marker.scale.x = self.controlling_vehicle.length
        self.ego_marker.scale.y = self.controlling_vehicle.width
        self.ego_marker.scale.z = 1.5
        self.ego_marker.pose.orientation.w = 1.0 # ToDo: add in orientation
        self.ego_marker.pose.position.x = self.controlling_vehicle.x
        self.ego_marker.pose.position.y = -self.controlling_vehicle.y
        self.ego_marker.pose.position.z = 0.75
        self.ego_marker.color.r = 1.0
        self.ego_marker.color.a = 1.0
        self.ego_marker.id = marker_id
        marker_id += 1

        # lanes
        self.lane_marker = MarkerArray()

        for l in self.lanes:
            for leng in range(0, l.length,5):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.a = 1.0
                marker.color.g = 1.0
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = l.starting_x + leng * np.cos(l.starting_theta)
                marker.pose.position.y = -(l.starting_y - leng * np.sin(l.starting_theta))
                marker.pose.position.z = 0
                marker.id=marker_id
                marker_id += 1
                self.lane_marker.markers.append(marker)
        self.lock.release()

    def renderScene(self):
        for v in self.vehicles:
            v.step(self.time_step)
        self.controlling_vehicle.step(self.time_step)
        self.updateMessages()
        self.updateMarkers()

    def resetScene(self, num_vehicles=[30, 0], num_lanes=2, lane_width_m=[3, 3], lane_length_m=500, \
                   max_vehicle_gaps_vehicle_len=5, min_vehicle_gaps_vehicle_len=0.5, \
                   vehicle_width=2, vehicle_length=4, starting_lane=-1, initial_speed=10):

        if (starting_lane < 0):
            self.cur_lane = num_lanes - 1
        else:
            self.cur_lane = starting_lane

        lane_y = 0
        max_vehicle_head_pos = vehicle_length
        for i in range(num_lanes):

            # add the lanes
            self.lanes.append(LaneSim(0, lane_y + lane_width_m[i] / 2, 0, lane_length_m, lane_width_m[i]))
            lane_y += lane_width_m[i]

            # add the vehicles
            num_veh = num_vehicles[i]
            prev_vehicle_head_pos = 0

            # ToDo: place vehicles on the lanes
            for j in range(num_veh):
                vehicle_gap = random.uniform(min_vehicle_gaps_vehicle_len, max_vehicle_gaps_vehicle_len)
                self.vehicles.append(Vehicle(vehicle_length, vehicle_width))
                self.vehicles[-1].place(prev_vehicle_head_pos + (vehicle_gap + 0.5) * vehicle_length,
                                        self.lanes[i].starting_y, self.lanes[i].starting_theta)
                self.vehicles[-1].setSpeed(initial_speed)
                prev_vehicle_head_pos += (vehicle_gap + 1) * vehicle_length
            max_vehicle_head_pos = max(max_vehicle_head_pos, prev_vehicle_head_pos)

        # current vehicle
        self.controlling_vehicle = Vehicle(vehicle_length, vehicle_width)
        #cur_vehicle_x = random.uniform(vehicle_length, max_vehicle_head_pos)
        cur_vehicle_x = 3
        cur_vehicle_y = self.lanes[self.cur_lane].starting_y
        cur_vehicle_theta = self.lanes[self.cur_lane].starting_theta
        self.controlling_vehicle.place(cur_vehicle_x, cur_vehicle_y, cur_vehicle_theta)
        self.controlling_vehicle.setSpeed(initial_speed)

    def publishFunc(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.publishMessages()
            self.renderScene()
            rate.sleep()


if __name__ == '__main__':
    try:
        simple_sim = SimpleSimulator(0.05)
        simple_sim.initialize()
        pub_thread = threading.Thread(target=simple_sim.publishFunc)
        pub_thread.start()
        simple_sim.spin()
    except rospy.ROSInterruptException:
        pass
