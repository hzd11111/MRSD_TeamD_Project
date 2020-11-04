import carla
import pickle
import sys
sys.path.append("../../carla_utils/utils")
from actors import Vehicle
from options import TrafficLightStatus

TOWN_O5_LIGHTS = 'traffic_lights/town05_traffic_light_x.pkl'
LIGHTS_DICT = TOWN_O5_LIGHTS

class TrafficLightManager():
    def __init__(self, client, town="Town05"):
        
        self.client = client
        self.world = self.client.get_world()
        self.Map = self.world.get_map()
        
        # a tick is needed to update the actors in the world
        self.world.tick()

        if town is "Town05":
            self.lane_id_to_tl_x_loc_dict = pickle.load(open(LIGHTS_DICT, "rb"))
        else:
            raise NotImplementedError
        
        self.all_traffic_lights = self.world.get_actors().filter("traffic.traffic_light*")
        self.road_to_tl_actor_dict = self.get_road_to_tl_actorid_dict()

    def get_road_to_tl_actorid_dict(self):
        dic = {}
        
        for tl in self.all_traffic_lights:
            x = tl.get_location().x
            for key, value in self.lane_id_to_tl_x_loc_dict.items():
                if x == value:
                    dic[key] = tl.id
        
        return dic
    
    def get_actor_to_traffic_light(self, actor: carla.Vehicle):
        # extract the actor's current road_id, and lane_id
        location = actor.get_location()
        nearest_waypoint = self.Map.get_waypoint(location, project_to_road=True)
        road_id = nearest_waypoint.road_id
        lane_id = nearest_waypoint.lane_id
        
        # get the traffic light from the stored dictionary
        tl = self.get_traffic_light_from_road(road_id=road_id, lane_id=lane_id)

        return tl, nearest_waypoint

    def set_actor_traffic_light_state(self, actor:Vehicle, is_ego=False):
        '''Set the traffic_light_status parameter for a Vehicle object'''
        # create a carla actor object from the Vehicle class
        carla_actor = self.world.get_actor(actor.actor_id)
        
        # get the traffic light state
        tl, nearest_waypoint = self.get_actor_to_traffic_light(carla_actor)
        
        if tl is None:
            actor.traffic_light_status = TrafficLightStatus.GREEN
        else:
            state = str(tl.get_state())

            # get the traffic manager 
            if state == 'Red':
                actor.traffic_light_status = TrafficLightStatus.RED
            elif state == 'Yellow':
                actor.traffic_light_status = TrafficLightStatus.YELLOW
            elif state == 'Green':
                actor.traffic_light_status = TrafficLightStatus.GREEN
        
        if is_ego and tl is not None:
            if state == 'Green':
                actor.traffic_light_stop_distance = -1
            else: 
                dist = len(nearest_waypoint.next_until_lane_end(1))
                actor.traffic_light_stop_distance = dist

    def set_traffic_light_for_vehicle(self, vehicle, raise_exception=False):
        carla_actor = self.world.get_actor(vehicle.actor_id)
        tl_state = carla_actor.get_traffic_light_state() 
        
        nearest_waypoint = self.Map.get_waypoint(carla_actor.get_location())
        road_id = nearest_waypoint.road_id
        lane_id = nearest_waypoint.lane_id

        tl = self.get_traffic_light_from_road(road_id, lane_id, raise_exception)

    def get_traffic_light_from_road(self, road_id, lane_id, raise_exception=False):
        '''
        Inputs:
            road_id (int): CARLA road_id
            lane_id (int): CARLA lane_id
            raise_exception (bool): flag to raise an exception is road_id, lane_id has
                                    no traffic light
        Returns:
            carla.Actor obj of the traffic light
        '''
        actor_id = self.road_to_tl_actor_dict.get((road_id, lane_id))

        if actor_id is not None:
            return self.world.get_actor(actor_id)
        else:
            if raise_exception:
                raise Exception("No traffic light found for this road_id," \
                            "lane_id combo", road_id, lane_id)
            return None

if __name__ == "__main__":
    '''Testing this script with manual_control.py'''
    # Updating traffic light status for each vehicle takes 0.08 ms

    # connect to client, get world
    client = carla.Client('127.0.0.1', 2000)
    world = client.get_world()

    # get ego vehicle spawned by manual control
    actor = world.get_actors().filter("vehicle*")[0]
    tlm = TrafficLightManager(client)
    
    # make a Vehicle object
    ac = Vehicle(world, actor.id)
    
    # keep printing ROS msg for the ego vehicle, check for light status
    while True:
        try:
            tlm.set_actor_traffic_light_state(ac, is_ego=True)
            print(ac.toRosMsg())
        except:
            import ipdb; ipdb.set_trace()
        import time; time.sleep(0.5)