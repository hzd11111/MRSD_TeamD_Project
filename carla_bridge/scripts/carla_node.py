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

# WILLBREAK fix pedestrian spelling, fix relative import path
from carla_utils.msg import ActorMsg, VehicleMsg, PedestrainMsg, FrenetMsg

sys.path.insert(1, '/home/grasp/Fall2020/src/carla_utils/utils')
from actors import *

from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import Pedestrian
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import PathPlan

from scenario_manager import CustomScenario
from grasp_path_planner.srv import SimService, SimServiceResponse
from agents.tools.misc import get_speed

from utils import *


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
        '''
        Creates Vehicle ROS msg and returns it.
        Inputs:
            actor (carla.Actor): carla actor object
        Returns: 
            vehicle_msg (VehicleMsg): ROS msg for vehicle state
        '''
        if actor == None:
            return None
        import ipdb; ipdb.set_trace()

        vehicle = Vehicle(world=actor.get_world(), actor_id=actor.id)
        vehicle_msg = vehicle.toRosMsg() 

        return vehicle_msg

    def getPedestrianState(self, actor, pedestrian_radius=0.5):
        '''Creates pedestrian State ROS msg'''
        #TODO: update this class to get Pedestrian msg instead of PEdesterianState msg
        ## TODO: Processing them like vehicles for now
        if actor == None:
            return None

        pedestrian = Pedestrian(world=actor.get_world(), actor_id=actor.id)
        pedestrian_msg = pedestrian.toRosMsg()

        return pedestrian_msg

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

    def distance_to_line(self, p1, p2, x, y):
        # TODO: Delete
        x_diff = p2[0] - p1[0]
        y_diff = p2[1] - p1[1]
        num = abs(y_diff * x - x_diff * y + p2[0] * p1[1] - p2[1] * p1[0])
        den = np.sqrt(y_diff ** 2 + x_diff ** 2)
        return num / float(den)

    def pathRequest(self, data):
        # data is pathplan ROS msg, check compatibility with the new msgs
        # TODO: make this compatible with the new ROS msg
        os.system("clear")

        reset_sim = False
        if self.first_frame_generated: # have we generated the first frame of 
            data = data.path_plan

            ### Get requested pose and speed values ###
            tracking_pose = data.tracking_pose
            tracking_speed = data.tracking_speed  # / 3.6
            reset_sim = data.reset_sim
            future_poses = data.future_poses

            ### Find distance from centre Line ###
            #TODO: make this a LaneStatus class or something or Vehicle class
            ego_location = self.ego_vehicle.get_location()
            ego_waypoint = self.carla_handler.world_map.get_waypoint(
                self.ego_vehicle.get_location(), project_to_road=True
            )
            next_waypoint = ego_waypoint.next(3)[0]
            prev_waypoint = ego_waypoint.previous(3)[0]

            point1 = [
                next_waypoint.transform.location.x,
                next_waypoint.transform.location.y,
            ]
            point2 = [
                prev_waypoint.transform.location.x,
                prev_waypoint.transform.location.y,
            ]
            lane_displacement = self.distance_to_line(
                point1, point2, ego_location.x, ego_location.y
            )
            speed_to_show = get_speed(self.ego_vehicle)

            # TODO: remove if not needed anymore, otherwise migrate to other class
            if self.tm.pedestrian_mode == False:
                distances2 = [
                    (
                        (
                            (ego_location.x - actor.get_location().x) ** 2
                            + (ego_location.y - actor.get_location().y) ** 2
                        ),
                        actor,
                    )
                    for actor in self.tm.world.get_actors().filter("vehicle.*")
                ]
                dists = [distances2[i][0] for i in range(len(distances2))]
                actors = [distances2[i][1] for i in range(len(distances2))]
                sorted_idx2 = np.argsort(dists)
                # closest_vehicle_distance = np.sqrt(dists[sorted_idx2[1]])
                closest_vehicle = actors[sorted_idx2[1]]

                bbox_ego = get_bounding_box(self.ego_vehicle)
                bbox_closest_vehicle = get_bounding_box(closest_vehicle)
                closest_distance = get_closest_distance(bbox_ego, bbox_closest_vehicle)

                if lane_displacement < 0.05:
                    print(
                        "Lane marking displacement:",
                        1.75
                        - self.ego_vehicle.bounding_box.extent.x / 2.0
                        - lane_displacement,
                        "| Current Speed:",
                        speed_to_show,
                        " | Distance to closest vehicle:",
                        closest_distance,
                    )
                    self.metrics.update(
                        closest_distance,
                        1.75
                        - self.ego_vehicle.bounding_box.extent.x / 2.0
                        - lane_displacement,
                        [ego_location.x, ego_location.y],
                    )
                else:
                    print(
                        "Lane marking displacement (Switching Lanes):",
                        lane_displacement,
                        "| Current Speed:",
                        speed_to_show,
                        " | Distance to closest vehicle:",
                        closest_distance,
                    )
                    self.metrics.update(
                        closest_distance,
                        lane_displacement,
                        [ego_location.x, ego_location.y],
                        ignore_lane=True,
                    )
            #-------------------------------------------------------------------

            # TODO: Remove
            for future_pose in future_poses:
                future_loc = carla.Location(
                    x=future_pose.x,
                    y=future_pose.y,
                    z=self.ego_vehicle.get_location().z,
                )
                self.carla_handler.world.debug.draw_string(
                    future_loc,
                    "O",
                    draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time=0.05,
                    persistent_lines=True,
                )

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
                    else:
                        if self.pedestrian_wait_frames != 0:
                            self.pedestrian_wait_frames -= 1
                        else:
                            # self.tm.pedestrian_controller.cross_road()
                            ego_location = self.ego_vehicle.get_location()
                            pedestrian_location = self.pedestrian.get_location()

                            distances2 = [
                                (
                                    (
                                        (ego_location.x - actor.get_location().x) ** 2
                                        + (ego_location.y - actor.get_location().y) ** 2
                                    ),
                                    actor,
                                )
                                for actor in self.tm.world.get_actors().filter(
                                    "walker.*"
                                )
                            ]
                            dists = [distances2[i][0] for i in range(len(distances2))]
                            actors = [distances2[i][1] for i in range(len(distances2))]
                            sorted_idx2 = np.argsort(dists)
                            # closest_vehicle_distance = np.sqrt(dists[sorted_idx2[1]])
                            closest_vehicle = actors[sorted_idx2[0]]

                            bbox_ego = get_bounding_box(self.ego_vehicle)
                            bbox_closest_vehicle = get_bounding_box(closest_vehicle)
                            closest_distance = (
                                get_closest_distance(bbox_ego, bbox_closest_vehicle)
                                + 0.1
                            )

                            print(
                                "Speed:",
                                get_speed(self.ego_vehicle),
                                "| Closest Distance:",
                                closest_distance,
                                "| Lane Marking Displacement:",
                                lane_displacement,
                            )
                            self.metrics.update(
                                closest_distance,
                                lane_displacement,
                                [ego_location.x, ego_location.y],
                            )

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

        # TODO: ROHAN move this to pedestrian class
        pedestrians_on_current_road = self.carla_handler.get_pedestrian_information(
            self.ego_vehicle
        )

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

        for idx in _: # TODO: ROHAN: Bounding box visualize for pedestrians and vehicles
            tmp_vehicle = actors_in_left_lane[idx]
            tmp_transform = tmp_vehicle.get_transform()
            tmp_bounding_box = tmp_vehicle.bounding_box
            tmp_bounding_box.location += tmp_transform.location
            self.carla_handler.world.debug.draw_box(
                tmp_bounding_box,
                tmp_transform.rotation,
                life_time=0.05,
                color=carla.Color(r=128, g=0, b=128),
            )

        if len(_) == 0:
            current_lane_vehicles, _2 = self.getClosest(
                [self.getVehicleState(actor) for actor in actors_in_current_lane],
                vehicle_ego,
                self.max_num_vehicles,
            )
            for idx in _2:
                tmp_vehicle = actors_in_current_lane[idx]
                tmp_transform = tmp_vehicle.get_transform()
                tmp_bounding_box = tmp_vehicle.bounding_box
                tmp_bounding_box.location += tmp_transform.location

                self.carla_handler.world.debug.draw_box(
                    tmp_bounding_box,
                    tmp_transform.rotation,
                    life_time=0.05,
                    color=carla.Color(r=128, g=0, b=128),
                )

        # TODO: move this logic to RewardInfo Class
        reward_info = RewardInfo()
        reward_info.time_elapsed = self.timestamp
        reward_info.new_run = self.first_run
        reward_info.collision = self.collision_marker
        reward_info.action_progress = self.action_progress
        reward_info.end_of_action = self.end_of_action
        reward_info.path_planner_terminate = self.path_planner_terminate
        env_state.reward = reward_info

        ## Pedestrian # TODO: ROHAN move this logic to pedestrian class 
        if self.pedestrian is not None:
            env_state.nearest_pedestrian = self.getClosestPedestrian(
                [self.getPedestrianState(actor) for actor in [self.pedestrian]],
                vehicle_ego,
                1,
            )[0]
            tmp_transform = self.pedestrian.get_transform()
            tmp_bounding_box = self.pedestrian.bounding_box
            tmp_bounding_box.location += tmp_transform.location

            self.carla_handler.world.debug.draw_box(
                tmp_bounding_box,
                tmp_transform.rotation,
                life_time=0.05,
                color=carla.Color(r=128, g=0, b=128),
            )
        else:
            env_state.nearest_pedestrian = Pedestrian()
            env_state.nearest_pedestrian.exist = False

        return SimServiceResponse(env_state) #TODO:Update with new EnvDesc class 

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
        # os.system("clear")
        print("Run Metrics:") #TODO: Remove unnecessary stuff
        print(
            "Distance travelled in currect iteration:",
            self.metrics.curr_distance_travelled,
        )
        self.metrics.reset()

        if self.tm.pedestrian_mode == True:
            print("Avg Distance:", self.metrics.avg_distance_travelled)
        else:
            print("Avg Distance To Lane Change:", self.metrics.avg_distance_travelled)
        # print("Avg Min Displacement from Lane Marking:", self.metrics.avg_max_dist_lane)
        # print("Avg Min Distance to objects:", self.metrics.avg_min_dist_actor)

        # time.sleep(1)
        synchronous_master = True

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
            # time.sleep(1)
            self.original_lane = -3 #TODO: check and remove 

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

        # Initialize PID Controller
        self.vehicle_controller = GRASPPIDController(
            self.ego_vehicle,
            args_lateral={
                "K_P": 0.5,
                "K_D": 0,
                "K_I": 0,
                "dt": self.simulation_sync_timestep,
            },
            args_longitudinal={
                "K_P": 1,
                "K_D": 0.0,
                "K_I": 0.0,
                "dt": self.simulation_sync_timestep,
            },
        )

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

        # self.carla_handler.draw_waypoints(current_lane_waypoints, life_time=100)
        # self.carla_handler.draw_waypoints(left_lane_waypoints, life_time=100, color=True)

        pedestrians_on_current_road = self.carla_handler.get_pedestrian_information(
            self.ego_vehicle
        )

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
        env_state.reward = reward_info

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
