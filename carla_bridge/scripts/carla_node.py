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
            lane_width,
        ) = state_information
        vehicle_ego = Vehicle(self.carla_handler.world, self.ego_vehicle.id)

        # Current Lane
        if reset_sim == True:

            # Current Lane
            current_lane_linestring = get_path_linestring(current_lane_waypoints)
            for i, wp in enumerate(current_lane_waypoints):
                current_lane_waypoints[i].frenet_pose = Frenet(
                    x=current_lane_linestring.project(
                        Point(wp.global_pose.x, wp.global_pose.y)
                    ),
                    y=0,
                )
            self.lane_cur = CurrentLane(
                lane_vehicles=actors_in_current_lane,
                lane_points=current_lane_waypoints,
                crossing_pedestrain=[
                    Pedestrian(self.carla_handler.world, self.pedestrian.id)
                ]
                if self.pedestrian is not None
                else [],
                origin_global_pose=current_lane_waypoints[0].global_pose
                if len(current_lane_waypoints) != 0
                else Pose2D(),
            )
            lane_cur = copy.copy(self.lane_cur)

            # Left Lane
            for i, wp in enumerate(left_lane_waypoints):
                left_lane_waypoints[i].frenet_pose = lane_cur.GlobalToFrenet(
                    left_lane_waypoints[i].global_pose
                )

            self.lane_left = ParallelLane(
                lane_vehicles=Vehicle.getClosest(actors_in_left_lane, vehicle_ego, n=5)[
                    0
                ],
                lane_points=left_lane_waypoints,
                same_direction=True,
                left_to_the_current=True,
                adjacent_lane=True,
                lane_distance=lane_width,
                origin_global_pose=left_lane_waypoints[0].global_pose
                if len(left_lane_waypoints) != 0
                else Pose2D(),
            )
            lane_left = copy.copy(self.lane_left)

            # Right Lane
            for i, wp in enumerate(right_lane_waypoints):
                right_lane_waypoints[i].frenet_pose = lane_cur.GlobalToFrenet(
                    right_lane_waypoints[i].global_pose
                )

            self.lane_right = ParallelLane(
                lane_vehicles=Vehicle.getClosest(
                    actors_in_right_lane, vehicle_ego, n=5
                )[0],
                lane_points=right_lane_waypoints,
                same_direction=True,
                left_to_the_current=False,
                adjacent_lane=True,
                lane_distance=lane_width,
                origin_global_pose=right_lane_waypoints[0].global_pose
                if len(right_lane_waypoints) != 0
                else Pose2D(),
            )
            lane_right = copy.copy(self.lane_right)

        else:
            lane_cur = copy.copy(self.lane_cur)
            # Update Values
            lane_cur.lane_vehicles = actors_in_current_lane

            # Left Lane
            lane_left = copy.copy(self.lane_left)
            # Update Values
            lane_left.lane_vehicles = Vehicle.getClosest(
                actors_in_left_lane, vehicle_ego, n=5
            )[0]

            # Right Lane
            lane_right = copy.copy(self.lane_right)
            # Update Values
            lane_right.lane_vehicles = Vehicle.getClosest(
                actors_in_right_lane, vehicle_ego, n=5
            )[0]

        ego_vehicle_frenet_pose = lane_cur.GlobalToFrenet(vehicle_ego.location_global)

        # Update Frenet Coordinates: Vehicles
        lane_cur.lane_vehicles = self.update_frenet(
            ego_vehicle_frenet_pose, lane_cur.lane_vehicles, lane_cur
        )
        lane_left.lane_vehicles = self.update_frenet(
            ego_vehicle_frenet_pose, lane_left.lane_vehicles, lane_cur
        )
        lane_right.lane_vehicles = self.update_frenet(
            ego_vehicle_frenet_pose, lane_right.lane_vehicles, lane_cur
        )

        # Update Frenet Coordinates: LanePoints
        lane_point_curr = copy.deepcopy(lane_cur.lane_points)
        lane_cur.lane_points = self.update_frenet_lanepoints(
            ego_vehicle_frenet_pose, lane_point_curr
        )
        lane_point_left = copy.deepcopy(lane_left.lane_points)
        lane_left.lane_points = self.update_frenet_lanepoints(
            ego_vehicle_frenet_pose, lane_point_left
        )
        lane_point_right = copy.deepcopy(lane_right.lane_points)
        lane_right.lane_points = self.update_frenet_lanepoints(
            ego_vehicle_frenet_pose, lane_point_right
        )

        vehicle_ego.location_frenet = ego_vehicle_frenet_pose

        lane_cur.ego_offset = ego_vehicle_frenet_pose.x
        vehicle_ego.location_frenet.x = 0

        for v in lane_left.lane_vehicles:
            print(
                vehicle_ego.location_frenet.x,
                vehicle_ego.location_frenet.y,
                "||",
                lane_cur.GlobalToFrenet(v.location_global).x,
                lane_cur.GlobalToFrenet(v.location_global).y,
                "||",
                v.location_frenet.x,
                v.location_frenet.y,
                "||",
                v.location_global.x,
                v.location_global.y,
                "||",
                vehicle_ego.location_global.x,
                vehicle_ego.location_global.y,
            )

        # import ipdb

        # ipdb.set_trace()

        print("\n")
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
        env_desc.adjacent_lanes = [lane_left, lane_right]
        env_desc.next_intersection = []
        env_desc.speed_limit = self.speed_limit
        env_desc.reward_info = reward_info

        # import ipdb

        # ipdb.set_trace()
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

        self.tm = CustomScenario(self.client, self.carla_handler)

        # Reset Environment
        self.resetEnv()

        state_information = self.carla_handler.get_state_information_new(
            self.ego_vehicle, self.original_lane
        )

        vehicle_ego = Vehicle(self.carla_handler.world, self.ego_vehicle.id)

        (
            current_lane_waypoints,
            left_lane_waypoints,
            right_lane_waypoints,
            front_vehicle,
            rear_vehicle,
            actors_in_current_lane,
            actors_in_left_lane,
            actors_in_right_lane,
            lane_width,
        ) = state_information

        ##############################################################################################################
        # publish the first frame
        # Current Lane
        current_lane_linestring = get_path_linestring(current_lane_waypoints)
        for i, wp in enumerate(current_lane_waypoints):
            current_lane_waypoints[i].frenet_pose = Frenet(
                x=current_lane_linestring.project(
                    Point(wp.global_pose.x, wp.global_pose.y)
                ),
                y=0,
            )
        self.lane_cur = CurrentLane(
            lane_vehicles=actors_in_current_lane,
            lane_points=current_lane_waypoints,
            crossing_pedestrain=[
                Pedestrian(self.carla_handler.world, self.pedestrian.id)
            ]
            if self.pedestrian is not None
            else [],
            origin_global_pose=current_lane_waypoints[0].global_pose
            if len(current_lane_waypoints) != 0
            else Pose2D(),
        )

        # Left Lane
        left_lane_linestring = get_path_linestring(left_lane_waypoints)
        for i, wp in enumerate(left_lane_waypoints):
            left_lane_waypoints[i].frenet_pose = Frenet(
                x=left_lane_linestring.project(
                    Point(wp.global_pose.x, wp.global_pose.y)
                ),
                y=0,
            )
        self.lane_left = ParallelLane(
            lane_vehicles=Vehicle.getClosest(actors_in_left_lane, vehicle_ego, n=5)[0],
            lane_points=left_lane_waypoints,
            same_direction=True,
            left_to_the_current=True,
            adjacent_lane=True,
            lane_distance=lane_width,
            origin_global_pose=left_lane_waypoints[0].global_pose
            if len(left_lane_waypoints) != 0
            else Pose2D(),
        )

        # Right Lane
        right_lane_linestring = get_path_linestring(right_lane_waypoints)
        for i, wp in enumerate(right_lane_waypoints):
            right_lane_waypoints[i].frenet_pose = Frenet(
                x=right_lane_linestring.project(
                    Point(wp.global_pose.x, wp.global_pose.y)
                ),
                y=0,
            )
        self.lane_right = ParallelLane(
            lane_vehicles=Vehicle.getClosest(actors_in_right_lane, vehicle_ego, n=5)[0],
            lane_points=right_lane_waypoints,
            same_direction=True,
            left_to_the_current=False,
            adjacent_lane=True,
            lane_distance=lane_width,
            origin_global_pose=right_lane_waypoints[0].global_pose
            if len(right_lane_waypoints) != 0
            else Pose2D(),
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
