import numpy as np
import itertools
import sys
sys.path.append("../../carla_utils/utils")

# ROS Packages
from utility import EnvDesc, CurrentLane
# other packages
from settings import Scenario


class StateManager:
    """
    A class to encompass all the logic needed to create state embeddings
    """
    def __init__(self, event):
        self.event = event

    def appendVehicleState(self, env_state, vehicle_state):
        env_state.append(vehicle_state.vehicle_location.x)
        env_state.append(vehicle_state.vehicle_location.y)
        env_state.append(vehicle_state.vehicle_location.theta)
        env_state.append(vehicle_state.vehicle_speed)
        return

    def createLaneChangeState(self, env_desc):
        """
        Create a dictionary of states. They keys are "adjacent", "front", "back", "pedestrians", "lane_distance".
        Each value for the actor types is a numpy array of the size (n X 6), where is the number of actors of that type
        """
        # extract all vehicles from adjacent lanes
        # TODO: Should I choose vehicles only from the lane to which we are shifting or all adjacent lanes?
        lane_distance = 0
        dummy_vehicle = [1000, 1000, 0, 0]
        pedestrian_states = []
        adj_lane_vehicles_states = []
        ego_vehicle_state = [
                        env_desc.cur_vehicle_state.location_frenet.x,
                        env_desc.cur_vehicle_state.location_frenet.y,
                        env_desc.cur_vehicle_state.location_frenet.theta,
                        env_desc.cur_vehicle_state.speed]

        for lane in env_desc.adjacent_lanes:
            if lane.left_to_the_current and lane.adjacent_lane:
                # Add relevant vehicles
                for vehicle in lane.lane_vehicles:
                    # import ipdb; ipdb.set_trace()
                    vehicle_state = [vehicle.location_frenet.x,
                                     vehicle.location_frenet.y,
                                     vehicle.location_frenet.theta,
                                     vehicle.speed]
                    adj_lane_vehicles_states.append(vehicle_state)
                # Add relevant pedestrians
                for pedestrian in lane.crossing_pedestrain:
                    pedestrian_state = [pedestrian.location_frenet.x,
                                        pedestrian.location_frenet.y,
                                        pedestrian.location_frenet.theta,
                                        pedestrian.speed]
                    pedestrian_states.append(pedestrian_state)
                # Get the lane distance
                lane_distance = lane.lane_distance
                break

        if len(adj_lane_vehicles_states) < 5:
            dummy_vehicles = list(itertools.repeat(dummy_vehicle, 5 - len(adj_lane_vehicles_states)))
            adj_lane_vehicles_states += dummy_vehicles

        # extract front vehicle and back vehicle and pedestrian in current lane
        # TODO: Make sure vehicle in front or back are always present
        current_lane = env_desc.current_lane
        front_vehicle = current_lane.VehicleInFront()
        back_vehicle = current_lane.VehicleBehind()

        for pedestrian in current_lane.crossing_pedestrain:
            pedestrian_state = [pedestrian.location_frenet.x,
                                pedestrian.location_frenet.y,
                                pedestrian.location_frenet.theta,
                                pedestrian.speed]
            pedestrian_states.append(pedestrian_state)

        if len(pedestrian_states) < 5:
            dummy_vehicles = list(itertools.repeat(dummy_vehicle, 5 - len(pedestrian_states)))
            pedestrian_states += dummy_vehicles

        # import ipdb; ipdb.set_trace()
        front_vehicle_state = [front_vehicle.location_frenet.x,
                               front_vehicle.location_frenet.y,
                               front_vehicle.location_frenet.theta,
                               front_vehicle.speed]

        back_vehicle_state = [back_vehicle.location_frenet.x,
                              back_vehicle.location_frenet.y,
                              back_vehicle.location_frenet.theta,
                              back_vehicle.speed]

        # concatenate all the states and lane distance
        entire_state = ego_vehicle_state + \
            [coord for state in adj_lane_vehicles_states for coord in state] + \
            front_vehicle_state + \
            back_vehicle_state + \
            [coord for state in pedestrian_states for coord in state] + [lane_distance]

        assert(len(entire_state) == 53)
        return np.array(entire_state)

    def createLaneFollowingState(self, env_desc):
        lane_following_state = {}
        ego_vehicle_state = [0, 0, 0]

        # extract front vehicle and back vehicle current lane
        current_lane = CurrentLane(env_desc.current_lane)
        front_vehicle = current_lane.vehicleInFront()
        back_vehicle = current_lane.vehicleBehind()

        front_vehicle_state = [front_vehicle.location_frenet.x,
                               front_vehicle.location_frenet.y,
                               front_vehicle.location_frenet.theta]

        back_vehicle_state = [back_vehicle.location_frenet.x,
                              back_vehicle.location_frenet.y,
                              back_vehicle.location_frenet.theta]

        lane_following_state["front"] = np.array([ego_vehicle_state + front_vehicle_state])
        lane_following_state["back"] = np.array([ego_vehicle_state + back_vehicle_state])
        # TODO: DO I add all the pedestrians which are relevant ? Maybe too much information
        pedestrian_states_concatenated = []
        for pedestrian in env_desc.pedestrians:
            pedestrian_state = [pedestrian.location_frenet.x,
                                pedestrian.location_frenet.y,
                                pedestrian.location_frenet.theta]
            pedestrian_states_concatenated.append(ego_vehicle_state + pedestrian_state)
        lane_following_state["pedestrians"] = np.array(pedestrian_states_concatenated)

    def createPedestrianState(self, env_desc):
        """
        Create a pedestrian avoidance event state
        Args:
        :param env_desc: (EnvironmentState) a ROS message describing the environment
        :param local: a bool flag describing whether the local or absolute coordinates must be used.
        """
        # choose between local and absolute coordinate system
        # the state embedding contains the vehicle and the pedestrian
        # TODO: Need find the closest pedestrian? Do we even consider this case?
        raise NotImplementedError()

    def embedState(self, env_desc: EnvDesc, scenario: Scenario, local=False) -> np.ndarray:
        """
        Create a state embedding vector from message recieved
        Args:
        :param env_desc: (EnvironmentState) a ROS message describing the environment
        """
        # based on scenario create the embedding accordingly
        if self.event == Scenario.LANE_CHANGE:
            return self.createLaneChangeState(env_desc)
        elif self.event == Scenario.LANE_FOLLOWING:
            return self.createLaneFollowingState(env_desc)
        elif self.event == Scenario.PEDESTRIAN:
            return self.createPedestrianState(env_desc)
        else:
            return {}
