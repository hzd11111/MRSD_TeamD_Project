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

NUM_NEXT_LANE_VEHICLES = 5
LANE_DISCRETIZATION  = 0.1

class LaneSim:
    def __init__(self, starting_x, starting_y, starting_theta, length, width):
        self.starting_x = starting_x
        self.starting_y = starting_y
        self.starting_theta = starting_theta
        self.length = length
        self.width = width
        self.ros_lane = False
        self.lock = threading.Lock()

    def convert2ROS(self):
        self.lock.acquire()
        if self.ros_lane == False:
            self.ros_lane = Lane()
            for lane_len in np.arange(0,self.length,LANE_DISCRETIZATION):
                lane_pt = LanePoint()
                lane_pt.pose.x = self.starting_x + lane_len * np.cos(self.starting_theta)
                lane_pt.pose.y = -(self.starting_y - lane_len * np.sin(self.starting_theta))
                lane_pt.pose.theta = self.starting_theta
                lane_pt.width = self.width
                self.ros_lane.lane.append(lane_pt)
        self.lock.release()
        return self.ros_lane


class Vehicle:
    def __init__(self, length, width, lane_num):
        self.length = length
        self.width = width
        self.x = 0
        self.y = 0
        self.theta = 0
        self.speed = 0
        self.lane = lane_num

    def place(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def step(self, duration):
        self.x += self.speed * np.cos(self.theta) * duration
        self.y -= self.speed * np.sin(self.theta) * duration

    def setSpeed(self, speed):
        self.speed = speed

    def convert2ROS(self):

        def speed_conversion(sim_speed):
            return sim_speed * 3.6

        vehicle_state = VehicleState()
        vehicle_state.vehicle_location.x = self.x
        vehicle_state.vehicle_location.y = self.y
        vehicle_state.vehicle_location.theta = self.theta
        vehicle_state.length = self.length
        vehicle_state.width = self.width
        vehicle_state.vehicle_speed = speed_conversion(self.speed)


class SimpleSimulator:
    def __init__(self, time_step):
        self.vehicles = []
        self.controlling_vehicle = None
        self.lanes = []
        self.cur_lane = 0
        self.time_step = time_step
        self.timestamp = 0
        self.first_run = 1

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
        self.lock.acquire()

        if not msg.id == self.id_waiting:
            self.lock.release()
            return
        tracking_pose = msg.tracking_pose
        tracking_speed = msg.tracking_speed
        reset_sim = msg.reset_sim

        if reset_sim:
            self.resetScene()
        else:
            self.controlling_vehicle.theta = np.arctan2(-(tracking_pose.y - self.controlling_vehicle.y),\
                                                        tracking_pose.x - self.controlling_vehicle.x)
            self.controlling_vehicle.setSpeed(tracking_speed)
            self.renderScene()
            self.id_waiting = self.id

        self.id += 1
        if self.id > 100000:
            self.id = 0




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

    def vehicleAxisProjection(self, veh, axis):
        x_dim = veh.length/2.
        y_dim = veh.width/2.

        # local corners
        local_corners = []
        local_corners.append(np.array([x_dim, y_dim]))
        local_corners.append(np.array([x_dim, -y_dim]))
        local_corners.append(np.array([-x_dim, y_dim]))
        local_corners.append(np.array([-x_dim, -y_dim]))

        # global corners
        global_corners = []
        for local in local_corners:
            global_x = veh.x + local[0] * np.cos(veh.theta) + local[1] * np.sin(veh.theta)
            global_y = veh.y - local[0] * np.sin(veh.theta) + local[1] * np.cos(veh.theta)
            global_corners.append(np.array(global_x, global_y))

        # projection
        proj_val = []
        for c in global_corners:
            proj_val.append(np.dot(c, axis))

        # return the min and max values
        return np.array([min(proj_val), max(proj_val)])


    def projNoOverlap(self, veh1_pair, veh2_pair):
        return veh1_pair[0] > veh2_pair[1] or veh1_pair[1] < veh2_pair[0]

    def vehAxis(self, veh):
        norm_axis = []
        norm_axis.append(np.array(np.cos(veh.theta), -np.sin(veh.theta)))
        norm_axis.append(np.array(np.sin(veh.theta), np.cos(veh.theta)))
        return norm_axis

    def vehicleToVehicleCollision(self, veh1, veh2):
        # normal axis
        norm_axis = []
        norm_axis.extend(self.vehAxis(veh1))
        norm_axis.extend(self.vehAxis(veh2))

        # collision check for each axis
        for ax in norm_axis:
            veh1_proj = self.vehicleAxisProjection(veh1, ax)
            veh2_proj = self.vehicleAxisProjection(veh2, ax)
            if self.projNoOverlap(veh1_proj, veh2_proj):
                return False
        return True

    def collisionCheck(self):
        for veh in self.vehicles:
            if self.vehicleToVehicleCollision(self.controlling_vehicle, veh):
                return True


    def updateMessages(self):
        self.lock.acquire()

        publish_msg = EnvironmentState()

        # current lane
        publish_msg.current_lane = self.lanes[self.cur_lane]

        # next lane
        next_lane = self.cur_lane - 1
        if next_lane >= 0:
            publish_msg.next_lane = self.lanes[next_lane]

        # vehicles
        closest_next_lane_vehicles = []
        front_veh = False
        back_veh = False

        def sort_veh(veh_tup):
            return veh_tup[1]

        veh_dir_vec = np.array(np.cos(self.controlling_vehicle.theta), -np.sin(self.controlling_vehicle.theta))
        for veh in self.vehicles:
            veh_dist = np.linalg.norm(np.array(veh.x - self.controlling_vehicle.x, veh.y - self.controlling_vehicle.y))

            if veh.lane == next_lane:
                # next lane vehicles
                if len(closest_next_lane_vehicles) < NUM_NEXT_LANE_VEHICLES:
                    closest_next_lane_vehicles.append(tuple((veh, veh_dist)))
                    closest_next_lane_vehicles.sort(key = sort_veh)
                elif veh_dist < closest_next_lane_vehicles[-1][1]:
                    closest_next_lane_vehicles[-1] = tuple((veh, veh_dist))
                    closest_next_lane_vehicles.sort(key = sort_veh)
            if veh.lane == self.cur_lane:
                x_diff = veh.x - self.controlling_vehicle.x
                y_diff = veh.y - self.controlling_vehicle.y
                pos_diff = np.array(x_diff, y_diff)
                proj = np.dot(veh_dir_vec, pos_diff)

                if proj > 0:  # front vehicle
                    if front_veh:
                        if front_veh[1] > veh_dist:
                            front_veh = tuple((veh, veh_dist))
                    else:
                        front_veh = tuple((veh, veh_dist))
                else:         # back vehicle
                    if back_veh:
                        if back_veh[1] > veh_dist:
                            back_veh = tuple((veh, veh_dist))
                    else:
                        back_veh = tuple((veh, veh_dist))

        # next lane vehicle documentation
        for veh_tup in closest_next_lane_vehicles:
            publish_msg.adjacent_lane_vehicles.append(veh_tup[0].convert2ROS())

        # front vehicle
        if front_veh:
            publish_msg.front_vehicle_state = front_veh[0].convert2ROS()

        # back vehicle
        if back_veh:
            publish_msg.back_vehicle_state = back_veh[0].convert2ROS()

        # ego vehicle
        publish_msg.cur_vehicle_state = self.controlling_vehicle.convert2ROS()

        # max # vehicles
        publish_msg.max_num_vehicles = NUM_NEXT_LANE_VEHICLES

        # speed limit
        publish_msg.speed_limit = 20

        # reward info
        reward_info = RewardInfo()
        reward_info.time_elapsed = self.timestamp
        reward_info.new_run = self.first_run
        reward_info.collision = self.collisionCheck()
        publish_msg.reward = reward_info

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
        self.timestamp += self.time_step
        self.controlling_vehicle.step(self.time_step)
        self.updateMessages()
        self.updateMarkers()
        self.first_run = 0

    def resetScene(self, num_vehicles=[30, 0], num_lanes=2, lane_width_m=[3, 3], lane_length_m=500, \
                   max_vehicle_gaps_vehicle_len=5, min_vehicle_gaps_vehicle_len=0.5, \
                   vehicle_width=2, vehicle_length=4, starting_lane=-1, initial_speed=10):
        self.timestamp = 0
        self.first_run = 1
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
                self.vehicles.append(Vehicle(vehicle_length, vehicle_width, i))
                self.vehicles[-1].place(prev_vehicle_head_pos + (vehicle_gap + 0.5) * vehicle_length,
                                        self.lanes[i].starting_y, self.lanes[i].starting_theta)
                self.vehicles[-1].setSpeed(initial_speed)
                prev_vehicle_head_pos += (vehicle_gap + 1) * vehicle_length
            max_vehicle_head_pos = max(max_vehicle_head_pos, prev_vehicle_head_pos)

        # current vehicle
        self.controlling_vehicle = Vehicle(vehicle_length, vehicle_width, self.cur_lane)
        #cur_vehicle_x = random.uniform(vehicle_length, max_vehicle_head_pos)
        cur_vehicle_x = 3
        cur_vehicle_y = self.lanes[self.cur_lane].starting_y
        cur_vehicle_theta = self.lanes[self.cur_lane].starting_theta
        self.controlling_vehicle.place(cur_vehicle_x, cur_vehicle_y, cur_vehicle_theta)
        self.controlling_vehicle.setSpeed(initial_speed)
        self.renderScene()


    def publishFunc(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.publishMessages()
            #self.renderScene()
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
