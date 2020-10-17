import carla
import pickle

TOWN_O5_LIGHTS = 'traffic_lights/town05_traffic_light_x.pkl'

class TrafficLightManager():
    def __init__(self, client, town="Town05"):
        
        self.client = client
        self.world = self.client.get_world()
        self.Map = self.world.get_map()
        
        if town is "Town05":
            self.lane_id_to_tl_x_loc_dict = pickle.load(open(TOWN_O5_LIGHTS, "rb"))
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
            print("No traffic light found for this road_id," \
                            "lane_id combo", road_id, lane_id)
            if raise_exception:
                raise Exception("No traffic light found for this road_id," \
                            "lane_id combo", road_id, lane_id)
            return None

if __name__ == "__main__":
    client = carla.Client('127.0.0.1', 2000)

    tlm = TrafficLightManager(client)

    import ipdb; ipdb.set_trace()