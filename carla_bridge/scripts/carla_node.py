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

from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import Pedestrian
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import PathPlan

from scenario_manager import CustomScenario
from grasp_path_planner.srv import SimService, SimServiceResponse
from agents.tools.misc import get_speed

from utils import *

sys.path.append("../../carla_utils/utils")
from utility import RewardInfo, EnvDesc

# from actors import Actor, Vehicle, Pedestrian

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

        self.metrics = DataCollector()

    def getVehicleState(self, actor):
        """Creates Vehicle State ROS msg"""
        # TODO: update this class to get Vehicle msg instead of VehicleState msg
        if actor == None:
            return None

        vehicle = VehicleState()
        vehicle.vehicle_location.x = actor.get_transform().location.x
        vehicle.vehicle_location.y = actor.get_transform().location.y
        vehicle.vehicle_location.theta = (
            actor.get_transform().rotation.yaw * np.pi / 180
        )  # CHECK : Changed this to radians.
        vehicle.vehicle_speed = (
            np.sqrt(
                actor.get_velocity().x ** 2
                + actor.get_velocity().y ** 2
                + actor.get_velocity().z ** 2
            )
            * 3.6
        )

        vehicle_bounding_box = actor.bounding_box.extent
        vehicle.length = (
            vehicle_bounding_box.x * 2
        )  # TODO: Check this, do we need to multiply
        vehicle.width = vehicle_bounding_box.y * 2

        return vehicle

    def getPedestrianState(self, actor, pedestrian_radius=0.5):
        """Creates pedestrian State ROS msg"""
        # TODO: update this class to get Pedestrian msg instead of PEdesterianState msg
        ## TODO: Processing them like vehicles for now
        if actor == None:
            return None

        pedestrian_state = Pedestrian()
        pedestrian_state.exist = True
        pedestrian_state.pedestrian_location.x = actor.get_transform().location.x
        pedestrian_state.pedestrian_location.y = actor.get_transform().location.y
        pedestrian_state.pedestrian_location.theta = (
            actor.get_transform().rotation.yaw * np.pi / 180
        )  # CHECK : Changed this to radians.
        pedestrian_state.radius = pedestrian_radius
        pedestrian_state.pedestrian_acceleration = 0
        pedestrian_state.pedestrian_speed = (
            np.sqrt(
                actor.get_velocity().x ** 2
                + actor.get_velocity().y ** 2
                + actor.get_velocity().z ** 2
            )
            * 3.6
        )

        return pedestrian_state

    def getLanePoints(self, waypoints, flip=False):
        # TODO: Lane classes should generate this type of messages
        # Remove this function
        lane_cur = Lane()
        lane_points = []

        for i, waypoint in enumerate(waypoints):
            lane_point = LanePoint()
            lane_point.pose.y = waypoint.transform.location.y
            lane_point.pose.x = waypoint.transform.location.x
            lane_point.pose.theta = (
                waypoint.transform.rotation.yaw * np.pi / 180
            )  # CHECK : Changed this to radians.
            lane_point.width = 3.5  # TODO
            lane_points.append(lane_point)

        lane_cur.lane = lane_points

        return lane_cur

    def pathRequest(self, data):
        # data is pathplan ROS msg, check compatibility with the new msgs
        # TODO: make this compatible with the new ROS msg
        os.system("clear")

        reset_sim = False
        if self.first_frame_generated:  # have we generated the first frame of
            data = data.path_plan

            ### Get requested pose and speed values ###
            tracking_pose = data.tracking_pose
            tracking_speed = data.tracking_speed  # / 3.6
            reset_sim = data.reset_sim

            # TODO: remove if not needed anymore, otherwise migrate to other class
            # -------------------------------------------------------------------

            self.end_of_action = data.end_of_action
            self.action_progress = data.action_progress
            self.path_planner_terminate = data.path_planner_terminate

            ### Update ROS Sync ###

            if reset_sim:
                self.resetEnv()

            else:
                self.first_run = 0

                # time.sleep(1)
                ### Apply Control signal on the vehicle. Vehicle and controller spawned in resetEnv###
                self.ego_vehicle.apply_control(
                    self.vehicle_controller.run_step(tracking_speed, tracking_pose)
                )

                # TODO:ROHAN: Move this to Pedesterian Class
                # pedestrian = Pedestrian(....)
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

                # TODO: Do we need this?
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

        # TODO: Ensure state information is extracted from the right class
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

        # TODO: Use the new class functions instead of getLanePoints
        # Current Lane
        if reset_sim == True:
            self.lane_cur = self.getLanePoints(current_lane_waypoints)
            lane_cur = self.lane_cur
            # Left Lane
            self.lane_left = self.getLanePoints(left_lane_waypoints)
            lane_left = self.lane_left
            # Right Lane
            self.lane_right = self.getLanePoints(right_lane_waypoints)
            lane_right = self.lane_right

        else:
            lane_cur = self.lane_cur

            # Left Lane
            lane_left = self.lane_left

            # Right Lane
            lane_right = self.lane_right

        # Ego vehicle
        vehicle_ego = self.getVehicleState(self.ego_vehicle)

        # Front vehicle
        if front_vehicle == None:
            vehicle_front = vehicle_ego
        else:
            vehicle_front = self.getVehicleState(front_vehicle)

        # Rear vehicle
        if rear_vehicle == None:
            vehicle_rear = vehicle_ego
        else:
            vehicle_rear = self.getVehicleState(rear_vehicle)

        # TODO: migrate logic to Environment State class, move it after reward state defined
        # Contruct enviroment state ROS message
        env_state = EnvironmentState()
        env_state.cur_vehicle_state = vehicle_ego
        env_state.front_vehicle_state = vehicle_front
        env_state.back_vehicle_state = vehicle_rear
        env_state.current_lane = lane_cur
        env_state.next_lane = lane_left
        env_state.adjacent_lane_vehicles, _ = self.getClosest(
            [self.getVehicleState(actor) for actor in actors_in_left_lane],
            vehicle_ego,
            self.max_num_vehicles,
        )
        env_state.speed_limit = self.speed_limit

        # TODO: move this logic to RewardInfo Class
        reward_info = RewardInfo()
        reward_info.time_elapsed = self.timestamp
        reward_info.new_run = self.first_run
        reward_info.collision = self.collision_marker
        reward_info.action_progress = self.action_progress
        reward_info.end_of_action = self.end_of_action
        reward_info.path_planner_terminate = self.path_planner_terminate
        env_state.reward = reward_info.toRosMsg()

        ## Pedestrian # TODO: ROHAN move this logic to pedestrian class
        if self.pedestrian is not None:
            env_state.nearest_pedestrian = self.getClosestPedestrian(
                [self.getPedestrianState(actor) for actor in [self.pedestrian]],
                vehicle_ego,
                1,
            )[0]

        else:
            env_state.nearest_pedestrian = Pedestrian()
            env_state.nearest_pedestrian.exist = False

        return SimServiceResponse(env_state)  # TODO:Update with new EnvDesc class

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
        # print("collision lol")
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

    def getClosest(self, adjacent_lane_vehicles, ego_vehicle, n=5):
        # TODO: ROHAN move to the Vehicle class
        ego_x = ego_vehicle.vehicle_location.x
        ego_y = ego_vehicle.vehicle_location.y

        distances = [
            (
                (ego_x - adjacent_lane_vehicles[i].vehicle_location.x) ** 2
                + (ego_y - adjacent_lane_vehicles[i].vehicle_location.y) ** 2
            )
            for i in range(len(adjacent_lane_vehicles))
        ]
        sorted_idx = np.argsort(distances)[:n]

        return [adjacent_lane_vehicles[i] for i in sorted_idx], sorted_idx

    def getClosestPedestrian(self, pedestrians, ego_vehicle, n=1):
        # TODO: ROHAN move to the Pedestrian class
        ego_x = ego_vehicle.vehicle_location.x
        ego_y = ego_vehicle.vehicle_location.y

        distances = [
            (
                (ego_x - pedestrians[i].pedestrian_location.x) ** 2
                + (ego_y - pedestrians[i].pedestrian_location.y) ** 2
            )
            for i in range(len(pedestrians))
        ]
        sorted_idx = np.argsort(distances)[:n]

        return [pedestrians[i] for i in sorted_idx]

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

        self.current_lane = current_lane_waypoints
        self.left_lane = left_lane_waypoints

        ##############################################################################################################
        # publish the first frame
        # Current Lane
        self.lane_cur = self.getLanePoints(current_lane_waypoints)

        # Left Lane
        self.lane_left = self.getLanePoints(left_lane_waypoints)
        # Right Lane
        self.lane_right = self.getLanePoints(right_lane_waypoints)

        vehicle_ego = self.getVehicleState(self.ego_vehicle)

        # Front vehicle
        if front_vehicle == None:
            vehicle_front = VehicleState()  # vehicle_ego
        else:
            vehicle_front = self.getVehicleState(front_vehicle)

        # Rear vehicle
        if rear_vehicle == None:
            vehicle_rear = VehicleState()  # vehicle_ego
        else:
            vehicle_rear = self.getVehicleState(rear_vehicle)

        # Contruct enviroment state ROS message
        env_state = EnvironmentState()
        env_state.cur_vehicle_state = vehicle_ego
        env_state.front_vehicle_state = vehicle_front
        env_state.back_vehicle_state = vehicle_rear
        env_state.adjacent_lane_vehicles, _ = self.getClosest(
            [self.getVehicleState(actor) for actor in actors_in_left_lane],
            vehicle_ego,
            self.max_num_vehicles,
        )  # TODO : Only considering left lane for now. Need to make this more general
        env_state.current_lane = self.lane_cur
        env_state.next_lane = self.lane_left
        env_state.max_num_vehicles = self.max_num_vehicles
        env_state.speed_limit = 20

        ## Pedestrian
        if self.pedestrian is not None:
            env_state.nearest_pedestrian = self.getClosestPedestrian(
                [self.getPedestrianState(actor) for actor in [self.pedestrian]],
                vehicle_ego,
                1,
            )[0]
        else:
            env_state.nearest_pedestrian = Pedestrian()
            env_state.nearest_pedestrian.exist = False

        reward_info = RewardInfo()
        reward_info.time_elapsed = self.timestamp
        reward_info.new_run = self.first_run
        reward_info.collision = self.collision_marker
        env_state.reward = reward_info.toRosMsg()

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
