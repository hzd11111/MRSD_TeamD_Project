import numpy as np
from random import choice

#  from timeit import default_timer as timer

import gym
from gym import spaces

import pygame
from pygame.sprite import spritecollide

from gym_racer.envs.racer_car import RacerCar
from gym_racer.envs.racer_map import RacerMap

#  from gym_racer.envs.utils import getMyLogger


class RacerEnv(gym.Env):
    metadata = {"render.modes": ["human", "console"]}
    reward_range = (-float("inf"), float("inf"))
    # TODO how to advertise sensor_array_type properly?

    def __init__(
        self,
        sensor_array_type="lidar",
        render_mode="human",
        sensor_array_params=None,
        dir_step=3,
        speed_step=1,
    ):
        """
        """
        #  logg = getMyLogger(f"c.{__class__.__name__}.__init__")
        #  logg.info(f"Start init RacerEnv")

        self.dir_step = dir_step
        self.speed_step = speed_step
        self.sensor_array_type = sensor_array_type
        self.render_mode = render_mode
        self.sensor_array_params = sensor_array_params

        # racing field dimensions
        self.field_wid = 900
        self.field_hei = 900
        self.field_size = (self.field_wid, self.field_hei)

        # sidebar info dimensions
        self.sidebar_wid = 300
        self.sidebar_hei = self.field_hei
        self.sidebar_size = (self.sidebar_wid, self.sidebar_hei)

        # total dimensions
        self.total_wid = self.sidebar_wid + self.field_wid
        self.total_hei = self.field_hei
        self.total_size = (self.total_wid, self.total_hei)

        # setup pygame environment
        self._setup_pygame()

        # setup the car
        self.racer_car = RacerCar(
            dir_step=self.dir_step,
            speed_step=self.speed_step,
            sensor_array_type=self.sensor_array_type,
            render_mode=self.render_mode,
            sensor_array_params=self.sensor_array_params,
        )

        # add the car to the list of sprites to render
        self.allsprites = pygame.sprite.RenderPlain((self.racer_car))

        # setup the road
        self.racer_map = RacerMap(
            self.field_wid, self.field_hei, render_mode=self.render_mode
        )

        # finish setup pygame environment
        self._finish_setup_pygame()

        # Define action and observation space
        self._setup_action_obs_space()

        # MAYBE this is called by the user to get the first obs anyway
        self.reset()

    def step(self, action):
        """Perform the action

        left-rigth: change steering
        up-down: accelerate/brake
        combination of the above
        do nothing
 
        ----------
        This method steps the game forward one step
        Parameters
        ----------
        action : str
            MAYBE should be int anyway
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing the
                state of the environment.
            reward (float) :
                amount of reward achieved by the previous action.
            episode_over (bool) :
                whether it's time to reset the environment again.
            info (dict) :
                diagnostic information useful for debugging.
        """

        #  logg = getMyLogger(f"c.{__class__.__name__}.step")
        #  logg.info(f"Start env step, action: '{action}'")

        # update the car
        self.racer_car.step(action)

        # compute the reward for this action
        reward, done = self._compute_reward()

        # get collisions from sensor array
        self._collide_sensor_array()

        # analyze the collisions
        obs = self._analyze_collisions()

        # create recap of env state
        info = {
            "car_pos_x": self.racer_car.pos_x,
            "car_pos_y": self.racer_car.pos_y,
            "car_dir": self.racer_car.direction,
            "car_speed": self.racer_car.speed,
        }

        return obs, reward, done, info

    def reset(self):
        """Reset the state of the environment to an initial state
        """
        #  logg = getMyLogger(f"c.{__class__.__name__}.reset")
        #  logg.debug(f"Start reset")

        #  pick a random segment of the map and place the car there
        direction, pos_x, pos_y = choice(self.racer_map.seg_info)
        self.racer_car.reset(pos_x, pos_y, direction)

        # get collisions from sensor array
        self._collide_sensor_array()

        # analyze the collisions
        # print("Debugging here  in reset")
        return self._analyze_collisions()

    def render(self, mode="console", close=False, reward=None):
        """Render the environment to the screen
        """
        #  logg = getMyLogger(f"c.{__class__.__name__}.render")
        #  logg.debug(f"Start render")

        if mode == "human" and not self.render_mode == "human":
            info_str = f"You tried to render the env in '{mode}' mode,"
            info_str += f" but the env is in '{self.render_mode}' mode."
            info_str += f"\nCreate a new env if you want to render to screen."
            info_str += f"\nI'll render as 'console'."
            print(info_str)
            mode = "console"

        if mode == "human":
            # Draw Everything again, every frame
            # the background already has the road and sidebar template drawn
            self.screen.blit(self.background, (0, 0))

            # draw all moving sprites (the car) on the screen
            self.allsprites.draw(self.screen)
            # if you draw on the field you can easily leave a track
            #  allsprites.draw(field)

            # draw the sensor surface
            self._draw_sensor_array()
            self.screen.blit(self.sa_surf, (0, 0))

            # update the dynamic sidebar
            self._update_dynamic_sidebar(reward)

            # update the display
            pygame.display.flip()

        elif mode == "console":
            info_str = f"State of the env:"
            info_str += f" speed: {self.racer_car.speed}"
            info_str += f" dir: {self.racer_car.direction}"
            info_str += f"\tReward: {reward}"
            print(info_str)

        else:
            raise ValueError(f"Unknown render mode {mode}")

    def _setup_action_obs_space(self):
        """
        """
        # self.action_space = spaces.MultiDiscrete([3, 3])
        self.action_space = spaces.Discrete(9)
        if self.sensor_array_type == "diamond":
            HEIGHT = self.racer_car.viewfield_size
            WIDTH = self.racer_car.viewfield_size
            N_CHANNELS = 1
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8
            )

        elif self.sensor_array_type == "lidar":
            HEIGHT = self.racer_car.tot_ray_num
            N_CHANNELS = 1
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(HEIGHT, N_CHANNELS), dtype=np.uint8
            )

        else:
            raise ValueError(f"Unknown sensor_array_type {self.sensor_array_type}")

    def _compute_reward(self):
        """compute the reward for moving on the map
        """
        #  logg = getMyLogger(f"c.{__class__.__name__}._compute_reward")
        #  logg.debug(f"Start _compute_reward")

        # compute collision car/road
        #  start = timer()
        hits = spritecollide(self.racer_car, self.racer_map, dokill=False)
        #  end = timer()
        #  logg.debug(f"Time for sprite collisions {end-start:.6f} s")

        #  logg.debug(f"hitting {hits}")
        hit_directions = []
        hit_sid = []
        for segment in hits:
            #  logg.debug(f"hit segment with id {segment.s_id}")
            hit_directions.append(self.racer_map.seg_info[segment.s_id][0])
            hit_sid.append(segment.s_id)

        # out of the map
        if len(hit_directions) == 0:
            return 0, True

        # if it is in the map, check that is moving
        if self.racer_car.speed < 0.0001:
            return -1, False

        # too many hits, your road is weird, cap them at 2 segments
        elif len(hit_directions) > 2:
            #  logg.warn(f"Too many segments hit")
            hit_directions = hit_directions[:2]
            hit_sid = hit_sid[:2]

        # now hit_directions is either 1 or 2 elements long
        if len(hit_directions) == 1:
            mean_direction = hit_directions[0]
        else:
            # 135   90  45    140   95  50    130   85  40
            # 180       0     185       5     175       -5
            # 225   270 315   230   275 320   220   265 310
            # 270, 0 have mean 315 = (270+0+360)/2
            # 270, 180 have mean 225 = (270+180)/2
            # 0, 90 have mean 45 = (0+90)/2
            if abs(hit_directions[0] - hit_directions[1]) > 180:
                mean_direction = (sum(hit_directions) + 360) / 2
                if mean_direction >= 360:
                    mean_direction -= 360
            else:
                mean_direction = sum(hit_directions) / 2

        error = self.racer_car.direction - mean_direction
        if error < 0:
            error += 360
        if error > 180:
            error = 360 - error
        #  logg.debug(f"direction {self.racer_car.direction} has error of {error:.4f}")

        # error goes from 0 (good) to 180 (bad)
        reward = 90 - error
        # MAYBE a sigmoid-like shape

        # scale it from -1 to 1
        reward /= 90

        # make it proportional to speed squared
        reward *= self.racer_car.speed * self.racer_car.speed

        return reward, False

    def _collide_sensor_array(self):
        """get the sa for the current direction and collide it with the road
        """
        #  logg = getMyLogger(f"c.{__class__.__name__}._collide_sensor_array")
        #  logg.debug(f"Start _collide_sensor_array")

        # get the current sensor_array to use
        self.curr_sa = self.racer_car.get_current_sensor_array()
        #  logg.debug(f"shape curr_sa {self.curr_sa.shape}")

        # copy the shape of curr_sa, but with one channel
        m = self.curr_sa.shape[0]
        n = self.curr_sa.shape[1]
        self.sa_collisions = np.zeros((m, n), dtype=np.uint8)

        # need to collide the entire matrix
        if self.sensor_array_type == "diamond":
            for i, row in enumerate(self.curr_sa):
                for j, s_pos in enumerate(row):
                    #  logg.debug(f"s_pos {s_pos.shape} : {s_pos}")
                    # check that the pos is inside the field
                    if (0 <= s_pos[0] < self.field_wid) and (
                        0 <= s_pos[1] < self.field_hei
                    ):
                        # extract the value of the map (road[1] - noroad[0]) at that pos
                        self.sa_collisions[i, j] = self.racer_map.raw_map[
                            s_pos[0], s_pos[1]
                        ]

        # for the lidar, when the first 0 is found on a line,
        # there is no need to keep colliding along that ray
        elif self.sensor_array_type == "lidar":
            for i, row in enumerate(self.curr_sa):
                for j, s_pos in enumerate(row):
                    # check that the pos is inside the field
                    if (0 <= s_pos[0] < self.field_wid) and (
                        0 <= s_pos[1] < self.field_hei
                    ):
                        # extract the value of the map (road[1] - noroad[0]) at that pos
                        road = self.racer_map.raw_map[s_pos[0], s_pos[1]]
                        if road == 0:
                            break
                        self.sa_collisions[i, j] = road

        else:
            raise ValueError(f"Unknown sensor_array_type {self.sensor_array_type}")

    def _analyze_collisions(self):
        """parse the collision matrix into obs
        """
        #  logg = getMyLogger(f"c.{__class__.__name__}._analyze_collisions")
        #  logg.debug(f"Start _analyze_collisions")

        if self.sensor_array_type == "diamond":
            # return the entire matrix
            obs = self.sa_collisions

        elif self.sensor_array_type == "lidar":
            #  logg.debug(f"shape sa_collisions {self.sa_collisions.shape}")
            #  logg.debug(f"sa_collisions\n{self.sa_collisions}")

            self.zero_strip = np.zeros((self.sa_collisions.shape[0], 1), dtype=np.uint8)
            #  logg.debug(f"shape zero_strip {self.zero_strip.shape}")
            pad_collisions = np.hstack((self.sa_collisions, self.zero_strip))
            #  logg.debug(f"shape pad_collisions {pad_collisions.shape}")
            #  logg.debug(f"pad_collisions\n{pad_collisions}")

            obs = np.argmin(pad_collisions, axis=1).reshape(-1,1)
            #  logg.debug(f"shape obs {obs.shape}")
            #  logg.debug(f"obs: {obs}")

        else:
            raise ValueError(f"Unknown sensor_array_type {self.sensor_array_type}")
        
        return obs

    def _setup_pygame(self):
        """
        """
        if self.render_mode == "human":
            # start pygame
            pygame.init()
            self.screen = pygame.display.set_mode(self.total_size)
            pygame.display.set_caption("Racer")

            # create the background that will be redrawn each iteration
            self.background = pygame.Surface(self.total_size)
            self.background = self.background.convert()

            # Create the playing field
            self.field = pygame.Surface(self.field_size)
            self.field = self.field.convert()
            self.field.fill((0, 0, 0))

            # where the info will be
            self._setup_sidebar()

            # create the surface for the sensor_array
            self.sa_surf = pygame.Surface(self.field_size)
            self.sa_surf = self.sa_surf.convert()
            # black colors will not be blit
            black = (0, 0, 0)
            self.sa_surf.set_colorkey(black)

        elif self.render_mode == "console":
            pass

        else:
            raise ValueError(f"Unknown render mode {self.render_mode}")

    def _finish_setup_pygame(self):
        if self.render_mode == "human":
            # draw map on the field, it is static, so there is no need to redraw it every time
            self.racer_map.draw(self.field)

            # draw the field (with the map on it) on the background
            self.background.blit(self.field, (0, 0))

        elif self.render_mode == "console":
            pass

        else:
            raise ValueError(f"Unknown render mode {self.render_mode}")

    def _draw_sensor_array(self):
        """draw the sensor_array on a Surface
        """
        #  logg = getMyLogger(f"c.{__class__.__name__}._draw_sensor_array")
        #  logg.debug(f"Start _draw_sensor_array")

        black = (0, 0, 0)
        # reset the Surface
        self.sa_surf.fill(black)

        the_color = (0, 255, 0)
        the_second_color = (0, 0, 255)
        color = the_color
        the_size = 2
        for i, row in enumerate(self.curr_sa):
            for j, s_pos in enumerate(row):
                if not self.sa_collisions is None:
                    if self.sa_collisions[i, j] == 1:
                        color = the_second_color
                    else:
                        color = the_color
                pygame.draw.circle(self.sa_surf, color, s_pos, the_size)

    def _setup_sidebar(self):
        """
        """
        #  logg = getMyLogger(f"c.{__class__.__name__}._setup_sidebar")
        #  logg.info(f"Start _setup_sidebar")

        # setup fonts to display info
        self._setup_font()

        self.sidebar_back_color = (80, 80, 80)
        self.font_info_color = (255, 255, 255)

        self.side_space = 50

        # create the sidebar surface
        self.sidebar_surf = pygame.Surface(self.sidebar_size)
        self.sidebar_surf = self.sidebar_surf.convert()
        self.sidebar_surf.fill(self.sidebar_back_color)

        # add titles
        self.speed_text_hei = 200
        text_speed = self.main_font.render("Speed:", 1, self.font_info_color)
        textpos_speed = text_speed.get_rect(
            midleft=(self.side_space, self.speed_text_hei)
        )
        self.sidebar_surf.blit(text_speed, textpos_speed)

        self.direction_text_hei = 260
        text_direction = self.main_font.render("Direction:", 1, self.font_info_color)
        textpos_direction = text_direction.get_rect(
            midleft=(self.side_space, self.direction_text_hei)
        )
        self.sidebar_surf.blit(text_direction, textpos_direction)

        self.reward_text_hei = 320
        text_reward = self.main_font.render("Reward:", 1, self.font_info_color)
        textpos_reward = text_reward.get_rect(
            midleft=(self.side_space, self.reward_text_hei)
        )
        self.sidebar_surf.blit(text_reward, textpos_reward)

        # setup positions for dynamic info: blit the text on a secondary
        # surface, then blit that on the screen in the specified position
        self.speed_val_pos = self.sidebar_wid - self.side_space, self.speed_text_hei
        self.direction_val_pos = (
            self.sidebar_wid - self.side_space,
            self.direction_text_hei,
        )
        self.reward_val_pos = self.sidebar_wid - self.side_space, self.reward_text_hei

        # create the dynamic sidebar surface
        self.side_dyn_surf = pygame.Surface(self.sidebar_size)
        self.side_dyn_surf = self.side_dyn_surf.convert()
        black = (0, 0, 0)
        self.side_dyn_surf.fill(black)
        self.side_dyn_surf.set_colorkey(black)

        # draw the static sidebar on the background
        self.background.blit(self.sidebar_surf, (self.field_wid, 0))

    def _update_dynamic_sidebar(self, reward=None):
        """fill the info values in the sidebar
        """
        #  logg = getMyLogger(f"c.{__class__.__name__}._update_dynamic_sidebar")
        #  logg.info(f"Start _update_dynamic_sidebar with reward {reward}")

        # reset the Surface
        black = (0, 0, 0)
        self.side_dyn_surf.fill(black)

        # speed text
        text_info_speed = self.main_font.render(
            f"{self.racer_car.speed}", 1, self.font_info_color, self.sidebar_back_color,
        )
        textpos_speed_info = text_info_speed.get_rect(midright=self.speed_val_pos)
        self.side_dyn_surf.blit(text_info_speed, textpos_speed_info)

        # direction text
        text_info_direction = self.main_font.render(
            f"{self.racer_car.direction}",
            1,
            self.font_info_color,
            self.sidebar_back_color,
        )
        textpos_direction_info = text_info_direction.get_rect(
            midright=self.direction_val_pos
        )
        self.side_dyn_surf.blit(text_info_direction, textpos_direction_info)

        # reward text
        if not reward is None:
            reward_val = f"{reward:.2f}"
        else:
            reward_val = f"-"
        text_info_reward = self.main_font.render(
            reward_val, 1, self.font_info_color, self.sidebar_back_color,
        )
        textpos_reward_info = text_info_reward.get_rect(midright=self.reward_val_pos)
        self.side_dyn_surf.blit(text_info_reward, textpos_reward_info)

        # draw the filled surface
        self.screen.blit(self.side_dyn_surf, (self.field_wid, 0))

    def _setup_font(self):
        """
        """
        #  logg = getMyLogger(f"c.{__class__.__name__}._setup_font")
        #  logg.info(f"Start _setup_font")

        #  logg.debug(f"all fonts {pygame.font.get_fonts()}")
        #  logg.debug(f"default font {pygame.font.get_default_font()}")
        #  logg.debug(f"match font hack {pygame.font.match_font('hack')}")

        if not pygame.font:
            raise RuntimeError("You need fonts to put text on the screen")
        # create a new Font object (from a file if you want)
        #  self.main_font = pygame.font.Font(None, 36)
        #  self.main_font = pygame.font.Font(pygame.font.match_font("hack"), 16)
        self.main_font = pygame.font.SysFont("arial", 26)
