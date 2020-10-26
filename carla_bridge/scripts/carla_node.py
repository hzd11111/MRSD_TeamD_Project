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
from p2p_scenario_manager import P2PScenario
from intersection_scenario_manager import IntersectionScenario
from grasp_path_planner.srv import SimService, SimServiceResponse
from agents.tools.misc import get_speed

# from agents.navigation.local_planner import LocalPlanner
from agents.navigation.roaming_agent import RoamingAgent

from utils import *

sys.path.append("../../carla_bridge/scripts/cartesian_to_frenet")
sys.path.insert(0, "../../../global_route_planner/")
from global_planner import get_global_planner

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
    GlobalPathPoint,
    GlobalPath,
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
        self.next_intersection = None
        self.global_path = None
        self.road_lane_to_orientation = None
        self.all_vehicles = None

        ### Autopilot
        self.autopilot_recompute_flag = 0
        self.agent = None
        self.global_planner = None
        self.global_path_carla_waypoints = None

    def pathRequest(self, data):

        plan = PathPlan.fromRosMsg(data.path_plan)

        reset_sim = False
        if self.first_frame_generated:  # have we generated the first frame of

            ### Get requested pose and speed values ###
            tracking_pose = plan.tracking_pose
            tracking_speed = plan.tracking_speed  # / 3.6
            reset_sim = plan.reset_sim
            is_autopilot = plan.auto_pilot

            self.end_of_action = plan.end_of_action
            self.action_progress = plan.action_progress
            self.path_planner_terminate = plan.path_planner_terminate

            ### Update ROS Sync ###

            if reset_sim:
                self.resetEnv()

            else:
                self.first_run = 0

                if not is_autopilot:
                    self.autopilot_recompute_flag = 0
                    ### Apply Control signal on the vehicle. Vehicle and controller spawned in resetEnv###
                    self.ego_vehicle.apply_control(
                        self.vehicle_controller.run_step(tracking_speed, tracking_pose)
                    )
                else:
                    if self.autopilot_recompute_flag == 0:
                        destination_location = (
                            self.global_path_carla_waypoints[-1]
                            .next(5)[0]
                            .transform.location
                        )
                        ego_vehicle_location = self.ego_vehicle.get_transform().location

                        tmp_route = self.global_planner.trace_route(
                            ego_vehicle_location, destination_location
                        )
                        self.agent._local_planner.set_global_plan(tmp_route)
                        self.autopilot_recompute_flag = 1
                        print("Recomputing autopilot route...")

                    control = self.agent.run_step(debug=True)
                    self.ego_vehicle.apply_control(control)

                # self.draw_location(tracking_pose)
                speed = self.ego_vehicle.get_velocity()
                print("Speed:", np.linalg.norm([speed.x, speed.y, speed.z]) * 3.6)

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

        if self.lane_cur == None:
            state_information = self.carla_handler.get_state_information_intersection(
                self.ego_vehicle,
                self.all_vehicles,
                self.ego_start_road_lane_pair,
                self.intersection_topology,
                self.road_lane_to_orientation,
            )

        else:
            state_information = self.carla_handler.get_state_information_intersection(
                self.ego_vehicle,
                self.all_vehicles,
                self.ego_start_road_lane_pair,
                self.intersection_topology,
                self.road_lane_to_orientation,
                only_actors=True,
            )

        ego_nearest_waypoint = self.carla_handler.world_map.get_waypoint(
            self.ego_vehicle.get_location(), project_to_road=True
        )
        self.carla_handler.world.debug.draw_string(
            ego_nearest_waypoint.transform.location,
            "O",
            draw_shadow=False,
            color=carla.Color(r=0, g=255, b=0),
            life_time=1,
        )

        vehicle_ego = Vehicle(self.carla_handler.world, self.ego_vehicle.id)

        (
            intersecting_left_info,
            intersecting_right_info,
            parallel_same_dir_info,
            parallel_opposite_dir_info,
            ego_lane_info,
        ) = state_information

        if self.lane_cur == None:
            # Current Lane
            self.lane_cur = CurrentLane(
                lane_vehicles=ego_lane_info[0][0],
                lane_points=ego_lane_info[0][1],
                crossing_pedestrain=[],
                origin_global_pose=ego_lane_info[0][1][0].global_pose
                if len(ego_lane_info[0][1]) != 0
                else Pose2D(),
            )

            self.adjacent_lanes = []
            for elem in parallel_same_dir_info:

                (
                    left_turning_lane,
                    right_turning_lane,
                    left_to_the_current,
                    right_next_to_the_current,
                ) = self.carla_handler.get_positional_booleans(
                    elem[2],
                    self.ego_start_road_lane_pair,
                    self.intersection_connections,
                    self.intersection_topology,
                    True,
                )

                parallel_lane = ParallelLane(
                    lane_vehicles=Vehicle.getClosest(elem[0], vehicle_ego, n=5)[0],
                    lane_points=elem[1],
                    same_direction=True,
                    left_to_the_current=left_to_the_current,
                    adjacent_lane=right_next_to_the_current,
                    lane_distance=10,
                    origin_global_pose=elem[1][0].global_pose
                    if len(elem[1]) != 0
                    else Pose2D(),
                    left_turning_lane=left_turning_lane,
                    right_turning_lane=right_turning_lane,
                )

                self.adjacent_lanes.append(parallel_lane)

            for elem in parallel_opposite_dir_info:
                (
                    left_turning_lane,
                    right_turning_lane,
                    left_to_the_current,
                    right_next_to_the_current,
                ) = self.carla_handler.get_positional_booleans(
                    elem[2],
                    self.ego_start_road_lane_pair,
                    self.intersection_connections,
                    self.intersection_topology,
                    False,
                )
                parallel_lane = ParallelLane(
                    lane_vehicles=Vehicle.getClosest(elem[0], vehicle_ego, n=5)[0],
                    lane_points=elem[1],
                    same_direction=False,
                    left_to_the_current=left_to_the_current,
                    adjacent_lane=right_next_to_the_current,
                    lane_distance=10,
                    origin_global_pose=elem[1][0].global_pose
                    if len(elem[1]) != 0
                    else Pose2D(),
                    left_turning_lane=left_turning_lane,
                    right_turning_lane=right_turning_lane,
                )
                self.adjacent_lanes.append(parallel_lane)

            # adjacent_lanes = copy.copy(self.adjacent_lanes)

            self.next_intersection = []
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
                self.next_intersection.append(perpendicular_lane)

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
                self.next_intersection.append(perpendicular_lane)

        else:
            # Update current lane vehicles. TODO: Check
            self.lane_cur.lane_vehicles = ego_lane_info[0][0]
            for i, elem in enumerate(parallel_same_dir_info):
                self.adjacent_lanes[i].lane_vehicles = Vehicle.getClosest(
                    elem[0], vehicle_ego, n=5
                )[0]
            for j, elem in enumerate(parallel_opposite_dir_info):
                self.adjacent_lanes[i + j + 1].lane_vehicles = Vehicle.getClosest(
                    elem[0], vehicle_ego, n=5
                )[0]
            for i, elem in enumerate(intersecting_left_info):
                self.next_intersection[i].lane_vehicles = Vehicle.getClosest(
                    elem[0], vehicle_ego, n=5
                )[0]
            for j, elem in enumerate(intersecting_right_info):
                self.next_intersection[i + j + 1].lane_vehicles = Vehicle.getClosest(
                    elem[0], vehicle_ego, n=5
                )[0]

        ### Get copies of current_lane, adjacent lanes and intersecting lanes
        lane_cur = copy.copy(self.lane_cur)
        adjacent_lanes = copy.copy(self.adjacent_lanes)
        next_intersection = copy.copy(self.next_intersection)

        ### Get the frenet coordinate of the ego vehicle in the current lane.
        ego_vehicle_frenet_pose = lane_cur.GlobalToFrenet(vehicle_ego.location_global)

        ### Update the ego_offset for current lane
        lane_cur.ego_offset = ego_vehicle_frenet_pose.x

        ### Update the ego offsets for parallel lanes
        for i in range(len(adjacent_lanes)):
            if adjacent_lanes[i].same_direction:
                adjacent_lanes[i].ego_offset = ego_vehicle_frenet_pose.x
            else:
                adjacent_lanes[i].ego_offset = (
                    adjacent_lanes[i].linestring.length - ego_vehicle_frenet_pose.x
                )

        ### TODO: Update origin and offsets for perpendicular lanes.
        self.update_intersecting_distance_and_intersecting_lane_origins(
            lane_cur, next_intersection
        )
        ### Update lane distances for adjacent lanes.
        self.update_lane_distance(lane_cur, adjacent_lanes)

        ### Update distance to intersection for each intersecting lane.
        for i in range(len(next_intersection)):
            next_intersection[i].intersecting_distance -= ego_vehicle_frenet_pose.x

        ### Update frenet for ego vehicle (will be (0,x,x))
        vehicle_ego.location_frenet = lane_cur.GlobalToFrenet(
            vehicle_ego.location_global
        )

        ### Update local frenet for vehicles on current lane
        for i in range(len(lane_cur.lane_vehicles)):
            lane_cur.lane_vehicles[i].location_frenet = lane_cur.GlobalToFrenet(
                lane_cur.lane_vehicles[i].location_global
            )

        ### Update frenet for vehicles on adjacent lanes
        # Update frenet after adjusting origin for the lane
        for i in range(len(adjacent_lanes)):
            for j in range(len(adjacent_lanes[i].lane_vehicles)):
                adjacent_lanes[i].lane_vehicles[j].location_frenet = adjacent_lanes[
                    i
                ].GlobalToFrenet(adjacent_lanes[i].lane_vehicles[j].location_global)

        ### Update frenet for vehicles on next_intersection lanes
        for i in range(len(next_intersection)):
            for j in range(len(next_intersection[i].lane_vehicles)):
                next_intersection[i].lane_vehicles[
                    j
                ].location_frenet = next_intersection[i].GlobalToFrenet(
                    next_intersection[i].lane_vehicles[j].location_global
                )

        ### Update all waypoint frenet coordinates.
        # Update current lane points
        for i in range(len(lane_cur.lane_points)):
            lane_cur.lane_points[i].frenet_pose = lane_cur.GlobalToFrenet(
                lane_cur.lane_points[i].global_pose
            )
        # Update points for adjacent lanes.
        for i in range(len(adjacent_lanes)):
            for j in range(len(adjacent_lanes[i].lane_points)):
                adjacent_lanes[i].lane_points[j].frenet_pose = adjacent_lanes[
                    i
                ].GlobalToFrenet(adjacent_lanes[i].lane_points[j].global_pose)
        # Update points for intersecting lanes.
        for i in range(len(next_intersection)):
            for j in range(len(next_intersection[i].lane_points)):
                next_intersection[i].lane_points[j].frenet_pose = next_intersection[
                    i
                ].GlobalToFrenet(next_intersection[i].lane_points[j].global_pose)

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
        env_desc.global_path = self.global_path

        return SimServiceResponse(env_desc.toRosMsg())

    def destroy_actors_and_sensors(self):

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

    def waypoint_to_pose2D(self, waypoint):

        x = waypoint.transform.location.x
        y = waypoint.transform.location.y
        theta = waypoint.transform.rotation.yaw * np.pi / 180

        return Pose2D(x=x, y=y, theta=theta)

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
                self.global_path_carla_waypoints,
                self.road_lane_to_orientation,
            ) = self.tm.reset(num_vehicles=10, junction_id=53, warm_start_duration=2)

            self.all_vehicles = self.carla_handler.world.get_actors().filter(
                "vehicle.*"
            )
            self.lane_cur = None
            self.adjacent_lanes = None
            self.next_intersection = None

            ## Initialize local planner

            self.agent = RoamingAgent(self.ego_vehicle)
            self.autopilot_recompute_flag = 0

            # self.local_planner.set_global_plan(route)
            # self.local_planner._min_distance = 20

            self.global_path = [
                self.waypoint_to_pose2D(wp)
                for wp in self.global_path_carla_waypoints[:-2]
            ]
            self.global_path = [
                GlobalPathPoint(global_pose=pose) for pose in self.global_path
            ]
            self.global_path = GlobalPath(path_points=self.global_path)

            self.draw_global_path(self.global_path)

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
                    "K_I": 0.1,
                    "dt": self.simulation_sync_timestep,
                },
            )
            self.ego_vehicle.apply_control(
                carla.VehicleControl(manual_gear_shift=True, gear=1)
            )
            self.ego_vehicle.apply_control(
                carla.VehicleControl(manual_gear_shift=False)
            )

        except rospy.ROSInterruptException:
            print("failed....")
            pass

    def initialize(self, synchronous_mode=True):
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

        if synchronous_mode:
            settings = self.carla_handler.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.simulation_sync_timestep
            self.carla_handler.world.apply_settings(settings)

        # self.tm = CustomScenario(self.client, self.carla_handler)

        self.tm = IntersectionScenario(self.client)

        self.global_planner = get_global_planner(
            world=self.carla_handler.world, planner_resolution=1
        )

        # Reset Environment
        self.resetEnv()

    def spin(self):
        print("Start Ros Spin")
        # spin
        rospy.spin()

    def update_intersecting_distance_and_intersecting_lane_origins(
        self, current_lane, perpendicular_lanes
    ):
        """
        Calculate the frenet coordinates of the vehicles in perpendicular lane wrt the current lane
        """
        for i in range(len(perpendicular_lanes)):
            intersecting_point = perpendicular_lanes[i].linestring.intersection(
                current_lane.linestring
            )
            perpendicular_lanes[
                i
            ].intersecting_distance = current_lane.linestring.project(
                intersecting_point
            )
            perpendicular_lanes[i].ego_offset = perpendicular_lanes[
                i
            ].linestring.project(intersecting_point)

    def update_lane_distance(self, current_lane, adjacent_lanes):
        """
        Calculate the lane_distance parameter for adjacent lanes.
        """
        for i in range(len(adjacent_lanes)):
            num_wps_in_lane = len(adjacent_lanes[i].lane_points)
            middle_lane_point = adjacent_lanes[i].lane_points[num_wps_in_lane // 2]
            middle_lane_point_global_pose = middle_lane_point.global_pose
            middle_point_frenet_relative_to_current_lane = current_lane.GlobalToFrenet(
                middle_lane_point_global_pose
            )
            adjacent_lanes[i].lane_distance = abs(
                middle_point_frenet_relative_to_current_lane.y
            )

    def draw(self, vehicle, color=(0, 255, 0)):

        vehicle_location = carla.Location(
            x=vehicle.location_global.x, y=vehicle.location_global.y, z=2
        )

        self.carla_handler.client.get_world().debug.draw_string(
            vehicle_location,
            "O",
            draw_shadow=False,
            color=carla.Color(r=color[0], g=color[1], b=color[2]),
            life_time=5,
        )

        print(vehicle.toRosMsg())

    def draw_location(self, location, color=(255, 0, 0)):

        location_ = carla.Location(location.x, location.y, z=2)

        self.carla_handler.client.get_world().debug.draw_string(
            location_,
            "O",
            draw_shadow=False,
            color=carla.Color(r=color[0], g=color[1], b=color[2]),
            life_time=1,
        )

    def draw_global_path(self, global_path, color=(0, 255, 0)):
        for point in global_path.path_points:
            location = carla.Location(x=point.global_pose.x, y=point.global_pose.y, z=2)

            self.carla_handler.client.get_world().debug.draw_string(
                location,
                "O",
                draw_shadow=False,
                color=carla.Color(r=color[0], g=color[1], b=color[2]),
                life_time=6,
            )


if __name__ == "__main__":
    try:
        carla_manager = CarlaManager()
        carla_manager.initialize(synchronous_mode=True)
        print("Initialize Done.....")
        carla_manager.spin()
    except rospy.ROSInterruptException:
        pass
