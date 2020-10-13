import time
import subprocess
import sys
import os

import rospy
import copy
import random
import threading
import copy

import carla

import agents.navigation.controller
import numpy as np
from shapely.geometry import Point

from carla_handler import CarlaHandler

from grasp_controller import GRASPPIDController

from scenario_manager import CustomScenario
from intersection_scenario_manager import IntersectionScenario
from grasp_path_planner.srv import SimService, SimServiceResponse
from agents.tools.misc import get_speed

from utils import *

sys.path.append("../../carla_bridge/scripts/cartesian_to_frenet")

from cartesian_to_frenet import (
    get_cartesian_from_frenet,
    get_frenet_from_cartesian,
    get_path_linestring,
)

sys.path.append("../../carla_utils/utils")
from utility import (
    RewardInfo,
    EnvDesc,
    CurrentLane,
    ParallelLane,
    PerpendicularLane,
    LanePoint,
    PathPlan,
)
from functional_utility import Pose2D, Frenet

from actors import Actor, Vehicle, Pedestrian

#######################################################################################

NODE_NAME = "carla_bridge"
SIM_TOPIC_NAME = "environment_state"
PATH_PLAN_TOPIC_NAME = "path_plan"

SIM_SERVICE_NAME = "simulator"

#######################################################################################


class CarlaManager:
    def __init__(self):

        ## CARLA Deps ##
        self.client = None
        self.carla_handler = None
        self.ego_vehicle = None
        self.vehicle_controller = None
        self.vehicles_list = []
        self.original_lane = None
        self.timestamp = 0
        self.collision_marker = 0
        self.first_run = 1
        self.lane_cur = None
        self.lane_left = None
        self.lane_right = None
        self.collision_sensor = None
        self.tm = None

        self.simulation_sync_timestep = 0.05
        self.first_frame_generated = False
        self.path_planner_terminate = False

        self.action_progress = 0
        self.end_of_action = True

        self.max_num_vehicles = 5

        self.speed_limit = None
        self.pedestrian = None
        self.pedestrian_wait_frames = 0

        ### Intersection Placeholders
        self.intersection_topology = None
        self.ego_start_road_lane_pair = None
        self.intersection_connections = None
        self.adjacent_lanes = None
        self.next_insersection = None

    def pathRequest(self, data):

        plan = PathPlan.fromRosMsg(data.path_plan)

        reset_sim = False
        if self.first_frame_generated:  # have we generated the first frame of

            ### Get requested pose and speed values ###
            tracking_pose = plan.tracking_pose
            tracking_speed = plan.tracking_speed  # / 3.6
            reset_sim = plan.reset_sim

            self.end_of_action = plan.end_of_action
            self.action_progress = plan.action_progress
            self.path_planner_terminate = plan.path_planner_terminate

            ### Update ROS Sync ###

            if reset_sim:
                self.resetEnv()

            else:
                self.first_run = 0

                ### Apply Control signal on the vehicle. Vehicle and controller spawned in resetEnv###
                self.ego_vehicle.apply_control(
                    self.vehicle_controller.run_step(tracking_speed, tracking_pose)
                )

                #### Check Sync ###
                flag = 0
                while flag == 0:
                    try:
                        self.carla_handler.world.tick()
                        flag = 1
                    except:
                        print(
                            "Missed Tick...................................................................................."
                        )
                        continue
                self.timestamp += self.simulation_sync_timestep
        else:
            self.first_frame_generated = True
            self.resetEnv()

        state_information = self.carla_handler.get_state_information_intersection(
            self.ego_vehicle, self.ego_start_road_lane_pair, self.intersection_topology
        )

        vehicle_ego = Vehicle(self.carla_handler.world, self.ego_vehicle.id)

        (
            intersecting_left_info,
            intersecting_right_info,
            parallel_same_dir_info,
            parallel_opposite_dir_info,
            ego_lane_info,
        ) = state_information

        # Current Lane
        # if reset_sim == True:

        # Current Lane

        self.lane_cur = CurrentLane(
            lane_vehicles=ego_lane_info[0][0],
            lane_points=ego_lane_info[0][1],
            crossing_pedestrain=[],
            origin_global_pose=ego_lane_info[0][1][0].global_pose
            if len(ego_lane_info[0][1]) != 0
            else Pose2D(),
        )
        lane_cur = copy.copy(self.lane_cur)

        self.adjacent_lanes = []
        for elem in parallel_same_dir_info:
            parallel_lane = ParallelLane(
                lane_vehicles=Vehicle.getClosest(elem[0], vehicle_ego, n=5)[0],
                lane_points=elem[1],
                same_direction=True,
                left_to_the_current=False,
                adjacent_lane=False,
                lane_distance=10,
                origin_global_pose=elem[1][0].global_pose
                if len(elem[1]) != 0
                else Pose2D(),
            )
            self.adjacent_lanes.append(parallel_lane)

        for elem in parallel_opposite_dir_info:
            parallel_lane = ParallelLane(
                lane_vehicles=Vehicle.getClosest(elem[0], vehicle_ego, n=5)[0],
                lane_points=elem[1],
                same_direction=False,
                left_to_the_current=False,
                adjacent_lane=False,
                lane_distance=10,
                origin_global_pose=elem[1][0].global_pose
                if len(elem[1]) != 0
                else Pose2D(),
            )
            self.adjacent_lanes.append(parallel_lane)

        adjacent_lanes = copy.copy(self.adjacent_lanes)

        self.next_insersection = []
        for elem in intersecting_left_info:
            perpendicular_lane = PerpendicularLane(
                lane_vehicles=Vehicle.getClosest(elem[0], vehicle_ego, n=5)[0],
                lane_points=elem[1],
                intersecting_distance=10,
                directed_right=False,
                origin_global_pose=elem[1][0].global_pose
                if len(elem[1]) != 0
                else Pose2D(),
            )
            self.next_insersection.append(perpendicular_lane)

        for elem in intersecting_right_info:
            perpendicular_lane = PerpendicularLane(
                lane_vehicles=Vehicle.getClosest(elem[0], vehicle_ego, n=5)[0],
                lane_points=elem[1],
                intersecting_distance=10,
                directed_right=True,
                origin_global_pose=elem[1][0].global_pose
                if len(elem[1]) != 0
                else Pose2D(),
            )
            self.next_insersection.append(perpendicular_lane)

        next_intersection = copy.copy(self.next_insersection)

        # ego_vehicle_frenet_pose = lane_cur.GlobalToFrenet(vehicle_ego.location_global)

        # # Update Frenet Coordinates: Vehicles
        # lane_cur.lane_vehicles = self.update_frenet(
        #     ego_vehicle_frenet_pose, lane_cur.lane_vehicles, lane_cur
        # )
        # lane_left.lane_vehicles = self.update_frenet(
        #     ego_vehicle_frenet_pose, lane_left.lane_vehicles, lane_cur
        # )
        # lane_right.lane_vehicles = self.update_frenet(
        #     ego_vehicle_frenet_pose, lane_right.lane_vehicles, lane_cur
        # )

        # # Update Frenet Coordinates: LanePoints
        # lane_point_curr = copy.deepcopy(lane_cur.lane_points)
        # lane_cur.lane_points = self.update_frenet_lanepoints(
        #     ego_vehicle_frenet_pose, lane_point_curr
        # )
        # lane_point_left = copy.deepcopy(lane_left.lane_points)
        # lane_left.lane_points = self.update_frenet_lanepoints(
        #     ego_vehicle_frenet_pose, lane_point_left
        # )
        # lane_point_right = copy.deepcopy(lane_right.lane_points)
        # lane_right.lane_points = self.update_frenet_lanepoints(
        #     ego_vehicle_frenet_pose, lane_point_right
        # )

        # vehicle_ego.location_frenet = ego_vehicle_frenet_pose

        # lane_cur.ego_offset = ego_vehicle_frenet_pose.x
        # vehicle_ego.location_frenet.x = 0

        # import ipdb

        # ipdb.set_trace()

        self.speed_limit = 25

        # print("\n")
        reward_info = RewardInfo()
        reward_info.time_elapsed = self.timestamp
        reward_info.new_run = self.first_run
        reward_info.collision = self.collision_marker
        reward_info.action_progress = self.action_progress
        reward_info.end_of_action = self.end_of_action
        reward_info.path_planner_terminate = self.path_planner_terminate

        env_desc = EnvDesc()
        env_desc.cur_vehicle_state = vehicle_ego
        env_desc.current_lane = lane_cur
        env_desc.adjacent_lanes = adjacent_lanes
        env_desc.next_intersection = next_intersection
        env_desc.speed_limit = self.speed_limit
        env_desc.reward_info = reward_info
        return SimServiceResponse(env_desc.toRosMsg())

    def destroy_actors_and_sensors(self):
        # TODO: ROHAN: get destroy method from Actor Class
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()

        for actor in self.tm.world.get_actors().filter("vehicle.*"):
            actor.destroy()

        for actor in self.tm.world.get_actors().filter("walker.*"):
            actor.destroy()

        self.vehicles_list = []
        print("All actors destroyed..\n")

    def collision_handler(self, event):
        self.collision_marker = 1

    def update_frenet(self, ego_frenet, vehicle_list, current_lane):
        for i in range(len(vehicle_list)):
            vehicle_list[i].location_frenet = vehicle_list[i].fromControllingVehicle(
                ego_frenet, current_lane
            )
        return vehicle_list

    def update_frenet_lanepoints(self, ego_frenet, lanepoints):
        for i in range(len(lanepoints)):
            lanepoints[i].frenet_pose.x = lanepoints[i].frenet_pose.x - ego_frenet.x
        return lanepoints

    def resetEnv(self):

        self.destroy_actors_and_sensors()
        self.timestamp = 0
        self.collision_marker = 0
        self.first_run = 1

        try:

            (
                self.ego_vehicle,
                self.vehicles_list,
                self.intersection_connections,
                self.intersection_topology,
                self.ego_start_road_lane_pair,
            ) = self.tm.reset(num_vehicles=10, junction_id=53, warm_start_duration=5)
            ## Handing over control
            del self.collision_sensor
            self.collision_sensor = self.carla_handler.world.spawn_actor(
                self.carla_handler.world.get_blueprint_library().find(
                    "sensor.other.collision"
                ),
                carla.Transform(),
                attach_to=self.ego_vehicle,
            )

            self.collision_sensor.listen(lambda event: self.collision_handler(event))
            self.vehicle_controller = GRASPPIDController(
                self.ego_vehicle,
                args_lateral={
                    "K_P": 0.5,
                    "K_D": 0,
                    "K_I": 0,
                    "dt": self.simulation_sync_timestep,
                },
                args_longitudinal={
                    "K_P": 3.5,
                    "K_D": 0,
                    "K_I": 0.01,
                    "dt": self.simulation_sync_timestep,
                },
            )

        except rospy.ROSInterruptException:
            print("failed....")
            pass

    def initialize(self):
        # initialize node
        rospy.init_node(NODE_NAME, anonymous=True)

        # initialize service
        self.planner_service = rospy.Service(
            SIM_SERVICE_NAME, SimService, self.pathRequest
        )

        # Start Client. Make sure Carla server is running before starting.

        client = carla.Client("localhost", 2000)
        client.set_timeout(2.0)
        print("Connection to CARLA server established!")

        # Create a CarlaHandler object. CarlaHandler provides some cutom built APIs for the Carla Server.
        self.carla_handler = CarlaHandler(client)
        self.client = client

        settings = self.carla_handler.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.simulation_sync_timestep
        self.carla_handler.world.apply_settings(settings)

        # self.tm = CustomScenario(self.client, self.carla_handler)

        self.tm = IntersectionScenario(self.client)

        # Reset Environment
        self.resetEnv()

        # state_information = self.carla_handler.get_state_information_intersection(
        #     self.ego_vehicle, self.ego_start_road_lane_pair, self.intersection_topology
        # )

        # vehicle_ego = Vehicle(self.carla_handler.world, self.ego_vehicle.id)

        # (
        #     intersecting_left_info,
        #     intersecting_right_info,
        #     parallel_same_dir_info,
        #     parallel_opposite_dir_info,
        #     ego_lane_info,
        # ) = state_information

        ##############################################################################################################
        # publish the first frame
        # Current Lane

        # self.lane_cur = CurrentLane(
        #     lane_vehicles=ego_lane_info[0][0],
        #     lane_points=ego_lane_info[0][1],
        #     crossing_pedestrain=[],
        #     origin_global_pose=ego_lane_info[0][1][0].global_pose
        #     if len(ego_lane_info[0][1]) != 0
        #     else Pose2D(),
        # )

        # self.adjacent_lanes = []
        # for elem in parallel_same_dir_info:
        #     parallel_lane = ParallelLane(
        #         lane_vehicles=Vehicle.getClosest(elem[0], vehicle_ego, n=5)[0],
        #         lane_points=elem[1],
        #         same_direction=True,
        #         left_to_the_current=False,
        #         adjacent_lane=False,
        #         lane_distance=10,
        #         origin_global_pose=elem[1][0].global_pose
        #         if len(elem[1]) != 0
        #         else Pose2D(),
        #     )
        #     self.adjacent_lanes.append(parallel_lane)

        # for elem in parallel_opposite_dir_info:
        #     parallel_lane = ParallelLane(
        #         lane_vehicles=Vehicle.getClosest(elem[0], vehicle_ego, n=5)[0],
        #         lane_points=elem[1],
        #         same_direction=False,
        #         left_to_the_current=False,
        #         adjacent_lane=False,
        #         lane_distance=10,
        #         origin_global_pose=elem[1][0].global_pose
        #         if len(elem[1]) != 0
        #         else Pose2D(),
        #     )
        #     self.adjacent_lanes.append(parallel_lane)

        # self.next_insersection = []
        # for elem in intersecting_left_info:
        #     perpendicular_lane = PerpendicularLane(
        #         lane_vehicles=Vehicle.getClosest(elem[0], vehicle_ego, n=5)[0],
        #         lane_points=elem[1],
        #         intersecting_distance=10,
        #         directed_right=False,
        #         origin_global_pose=elem[1][0].global_pose
        #         if len(elem[1]) != 0
        #         else Pose2D(),
        #     )
        #     self.next_insersection.append(perpendicular_lane)

        # for elem in intersecting_right_info:
        #     perpendicular_lane = PerpendicularLane(
        #         lane_vehicles=Vehicle.getClosest(elem[0], vehicle_ego, n=5)[0],
        #         lane_points=elem[1],
        #         intersecting_distance=10,
        #         directed_right=True,
        #         origin_global_pose=elem[1][0].global_pose
        #         if len(elem[1]) != 0
        #         else Pose2D(),
        #     )
        #     self.next_insersection.append(perpendicular_lane)

    def spin(self):
        print("Start Ros Spin")
        # spin
        rospy.spin()


if __name__ == "__main__":
    try:
        carla_manager = CarlaManager()
        carla_manager.initialize()
        print("Initialize Done.....")
        carla_manager.spin()
    except rospy.ROSInterruptException:
        pass
