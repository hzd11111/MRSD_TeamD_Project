import numpy as np

# ROS Packages
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import VehicleState
# other packages
from settings import Scenario


# TODO: Figure out if exactly how the heading works
def convertToLocal(cur_vehicle: VehicleState, adj_vehicle: VehicleState):
    """
    Converts the adj_vehicle position into cur_vehicle position
    Expects VehicleState objects and Euclidean Coordinate Frame

    Args
    :param cur_vehicle: (VehicleState) the ego vehicle state
    :param adj_vehicle: (VehicleState) the target vehicle state

    Returns
    the adj_vehicle pose in terms of the cur_vehicle
    """
    # create a place holder vehicle object to represent the resultant relative pose
    result_state = VehicleState()

    # extract the pose and velocities of the adj_vehicle
    x = adj_vehicle.vehicle_location.x
    y = adj_vehicle.vehicle_location.y
    theta = adj_vehicle.vehicle_location.theta
    speed = adj_vehicle.vehicle_speed

    # heading is in terms of the x-axis so vx is cos component
    # vx = speed * np.cos(theta)
    # vy = speed * np.sin(theta)

    # get current_vehicle_speeds
    # cvx = cur_vehicle.vehicle_speed * np.cos(cur_vehicle.vehicle_location.theta)
    # cvy = cur_vehicle.vehicle_speed * np.sin(cur_vehicle.vehicle_location.theta)

    # make homogeneous transform matrix to transform global coordinate into the frame
    # of the ego vehicle state. Calculate as a product of two transformations. Multiply right.
    H_Rot = np.eye(3)
    H_Rot[-1, -1] = 1
    H_Rot[0, -1] = 0
    H_Rot[1, -1] = 0
    H_Rot[0, 0] = np.cos(cur_vehicle.vehicle_location.theta)
    H_Rot[0, 1] = -np.sin(cur_vehicle.vehicle_location.theta)
    H_Rot[1, 0] = np.sin(cur_vehicle.vehicle_location.theta)
    H_Rot[1, 1] = np.cos(cur_vehicle.vehicle_location.theta)
    H_trans = np.eye(3)
    H_trans[0, -1] = -cur_vehicle.vehicle_location.x
    H_trans[1, -1] = -cur_vehicle.vehicle_location.y
    H = np.matmul(H_Rot, H_trans)
    # calculate and set relative position
    res = np.matmul(H, np.array([x, y, 1]).reshape(3, 1))
    result_state.vehicle_location.x = res[0, 0]
    result_state.vehicle_location.y = res[1, 0]
    # calculate and set relative orientation
    result_state.vehicle_location.theta = theta - cur_vehicle.vehicle_location.theta
    # calculate and set relative speed
    # res_vel = np.array([vx - cvx, vy - cvy])
    result_state.vehicle_speed = speed  # np.linalg.norm(res_vel)
    # time.sleep(5)
    return result_state


class StateManager:
    """
    A class to encompass all the logic needed to create state embeddings
    """
    def __init__(self, event):
        self.event = event

    def append_vehicle_state(self, env_state, vehicle_state):
        env_state.append(vehicle_state.vehicle_location.x)
        env_state.append(vehicle_state.vehicle_location.y)
        env_state.append(vehicle_state.vehicle_location.theta)
        env_state.append(vehicle_state.vehicle_speed)
        return

    def embedState(self, env_desc: EnvironmentState, scenario: Scenario, local=False) -> np.ndarray:
        """
        Create a state embedding vector from message recieved
        Args:
        :param env_des: (EnvironmentState) a ROS message describing the environment
        """
        # TODO: Convert the embedding to Frenet coordinate system
        # based on scenario create the embedding accordingly
        if self.event == Scenario.LANE_CHANGE:
            # initialize count of vehicles. We will add 5 vehicles
            # The lane change scenario contains the current vehicle and 5 adjacent vehicles in the state
            count_of_vehicles = 0
            env_state = []

            # if local flag is not set then use absolute coordinates
            if not local:
                self.append_vehicle_state(env_state, env_desc.cur_vehicle_state)
                # TODO: Currently not adding vehicles in the front and back. Might need to add these later.
                # self.append_vehicle_state(env_state, env_desc.back_vehicle_state)
                # self.append_vehicle_state(env_state, env_desc.front_vehicle_state)
                # count += 2
                # add vehicles in the adjacent lane
                for _, vehicle in enumerate(env_desc.adjacent_lane_vehicles):
                    if count_of_vehicles < 5:
                        self.append_vehicle_state(env_state, vehicle)
                    else:
                        break
                    count_of_vehicles += 1
            else:
                # use relative coordinates. Current vehicle is origin
                cur_vehicle_state = VehicleState()
                cur_vehicle_state.vehicle_location.x = 0
                cur_vehicle_state.vehicle_location.y = 0
                cur_vehicle_state.vehicle_location.theta = 0
                # Still choose absolute current_vehicle speed wrt global frame
                cur_vehicle_state.vehicle_speed = env_desc.cur_vehicle_state.vehicle_speed
                self.append_vehicle_state(env_state, cur_vehicle_state)

                # TODO: Currently not adding vehicles in the front and back. Might need to add these later.
                # add vehicles in the adjacent lane
                for _, vehicle in enumerate(env_desc.adjacent_lane_vehicles):
                    converted_state = convertToLocal(env_desc.cur_vehicle_state, vehicle)
                    if count_of_vehicles < 5:
                        self.append_vehicle_state(env_state, converted_state)
                    else:
                        break
                    count_of_vehicles += 1

            # add dummy vehicles into the state if not reached 5
            dummy = VehicleState()
            dummy.vehicle_location.x = 100
            dummy.vehicle_location.y = 100
            dummy.vehicle_location.theta = 0
            dummy.vehicle_speed = 0
            while count_of_vehicles < 5:
                self.append_vehicle_state(env_state, dummy)
                count_of_vehicles += 1
            return env_state

        elif self.event == Scenario.PEDESTRIAN:
            # initialize an empty list for the state
            env_state = []

            # choose between local and absolute coordinate system
            # the state embedding contains the vehicle and the pedestrian
            if not local:
                self.append_vehicle_state(env_state, env_desc.cur_vehicle_state)
                ped_vehicle = VehicleState()
                ped_vehicle.vehicle_location = env_desc.nearest_pedestrian.pedestrian_location
                ped_vehicle.pedestrian_speed = env_desc.nearest_pedestrian.pedestrian_speed
                self.append_vehicle_state(env_state, ped_vehicle)
            else:
                cur_vehicle_state = VehicleState()
                cur_vehicle_state.vehicle_location.x = 0
                cur_vehicle_state.vehicle_location.y = 0
                cur_vehicle_state.vehicle_location.theta = 0
                cur_vehicle_state.vehicle_speed = env_desc.cur_vehicle_state.vehicle_speed
                self.append_vehicle_state(env_state, cur_vehicle_state)
                ped_vehicle = VehicleState()
                ped_vehicle.vehicle_location = env_desc.nearest_pedestrian.pedestrian_location
                ped_vehicle.vehicle_speed = env_desc.nearest_pedestrian.pedestrian_speed
                converted_state = convertToLocal(env_desc.cur_vehicle_state, ped_vehicle)
                self.append_vehicle_state(env_state, converted_state)
            return env_state
