import time
import subprocess
import sys
import os

import rospy
import copy
import random
import threading

import carla

import agents.navigation.controller
import numpy as np

from carla_handler import CarlaHandler

from grasp_controller import GRASPPIDController

from scenario_manager import CustomScenario
from grasp_path_planner.srv import SimService, SimServiceResponse
from agents.tools.misc import get_speed

from utils import *

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
from functional_utility import Pose2D

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

    def pathRequest(self, data):

        os.system("clear")
        print(data)
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

                #### Pedestrian Spawning
                if self.tm.pedestrian_mode == True:
                    if self.pedestrian is None:
                        current_ego_speed = (
                            np.sqrt(
                                self.ego_vehicle.get_velocity().x ** 2
                                + self.ego_vehicle.get_velocity().y ** 2
                                + self.ego_vehicle.get_velocity().z ** 2
                            )
                            * 3.6
                        )
                        if current_ego_speed > 25:
                            self.tm.pedestrian_controller.waypoints_list = self.carla_handler.get_next_waypoints(
                                self.carla_handler.world_map.get_waypoint(
                                    self.ego_vehicle.get_location(),
                                    project_to_road=True,
                                ),
                                None,
                                k=45,
                            )[
                                self.tm.pedestrian_spawn_dist_low : self.tm.pedestrian_spawn_dist_high
                            ]
                            self.pedestrian = (
                                self.tm.pedestrian_controller.random_spawn()
                            )
                            self.tm.pedestrian_controller.cross_road()

                            print("Pedestrian Spawned")

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

        state_information = self.carla_handler.get_state_information_new(
            self.ego_vehicle, self.original_lane
        )
        (
            current_lane_waypoints,
            left_lane_waypoints,
            right_lane_waypoints,
            front_vehicle,
            rear_vehicle,
            actors_in_current_lane,
            actors_in_left_lane,
            actors_in_right_lane,
        ) = state_information

        # Current Lane
        if reset_sim == True:

            # Current Lane
            self.lane_cur = CurrentLane(
                lane_vehicles=actors_in_current_lane,
                lane_points=current_lane_waypoints,
                crossing_pedestrain=[
                    Pedestrian(self.carla_handler.world, self.pedestrian.id)
                ]
                if self.pedestrian is not None
                else [],
            )
            lane_cur = self.lane_cur

            # Left Lane
            self.lane_left = ParallelLane(
                lane_vehicles=actors_in_left_lane,
                lane_points=left_lane_waypoints,
                same_direction=True,
                left_to_the_current=True,
                adjacent_lane=True,
            )
            lane_left = self.lane_left
            # Right Lane
            self.lane_right = ParallelLane(
                lane_vehicles=actors_in_right_lane,
                lane_points=right_lane_waypoints,
                same_direction=True,
                left_to_the_current=False,
                adjacent_lane=True,
            )
            lane_right = self.lane_right

        else:
            lane_cur = self.lane_cur

            # Left Lane
            lane_left = self.lane_left

            # Right Lane
            lane_right = self.lane_right

        # Ego vehicle
        vehicle_ego = Vehicle(self.carla_handler.world, self.ego_vehicle.id)

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
        env_desc.adjacent_lanes = [self.lane_left, self.lane_right]
        env_desc.next_intersection = []
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

        if self.pedestrian is not None:
            self.tm.pedestrian_controller.destroy()
            self.pedestrian = None
            self.pedestrian_wait_frames = 10

        self.vehicles_list = []
        print("All actors destroyed..\n")

    def collision_handler(self, event):
        self.collision_marker = 1

    def resetEnv(self):

        self.destroy_actors_and_sensors()
        self.timestamp = 0
        self.collision_marker = 0
        self.first_run = 1

        try:

            self.ego_vehicle, self.vehicles_list, self.speed_limit = self.tm.reset()
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
                    "K_P": 0.5,
                    "K_D": 0,
                    "K_I": 0,
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

        self.tm = CustomScenario(self.client, self.carla_handler)

        # Reset Environment
        self.resetEnv()

        state_information = self.carla_handler.get_state_information_new(
            self.ego_vehicle, self.original_lane
        )

        ego_vehicle = Vehicle(self.carla_handler.world, self.ego_vehicle.id)

        (
            current_lane_waypoints,
            left_lane_waypoints,
            right_lane_waypoints,
            front_vehicle,
            rear_vehicle,
            actors_in_current_lane,
            actors_in_left_lane,
            actors_in_right_lane,
        ) = state_information

        ##############################################################################################################
        # publish the first frame
        # Current Lane
        self.lane_cur = CurrentLane(
            lane_vehicles=actors_in_current_lane,
            lane_points=current_lane_waypoints,
            crossing_pedestrain=[
                Pedestrian(self.carla_handler.world, self.pedestrian.id)
            ]
            if self.pedestrian is not None
            else [],
        )

        # Left Lane
        self.lane_left = ParallelLane(
            lane_vehicles=Vehicle.getClosest(actors_in_left_lane, ego_vehicle, n=5)[0],
            lane_points=left_lane_waypoints,
            same_direction=True,
            left_to_the_current=True,
            adjacent_lane=True,
        )

        # Right Lane
        self.lane_right = ParallelLane(
            lane_vehicles=Vehicle.getClosest(actors_in_right_lane, ego_vehicle, n=5)[0],
            lane_points=right_lane_waypoints,
            same_direction=True,
            left_to_the_current=False,
            adjacent_lane=True,
        )

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
