import numpy as np
from math import ceil
from math import cos
from math import radians
from math import sin

import pygame
from pygame import Surface
from pygame.sprite import Sprite
from pygame.transform import rotate

#  from gym_racer.envs.utils import getMyLogger
from gym_racer.envs.utils import compute_rot_matrix


class RacerCar(Sprite):
    def __init__(
        self,
        pos_x=0,
        pos_y=0,
        direction=0,
        dir_step = 3,
        speed_step = 1,
        sensor_array_type="lidar",
        render_mode="human",
        sensor_array_params=None,
    ):
        #  logg = logging.getLogger(f"c.{__name__}.__init__")
        #  logg.info(f"Start init RacerCar")

        super().__init__()

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.precise_x = pos_x
        self.precise_y = pos_y

        self.sensor_array_type = sensor_array_type
        self.render_mode = render_mode
        self.sensor_array_params = sensor_array_params

        self.direction = direction  # in degrees
        self.dir_step = dir_step

        self.speed = 0
        self.speed_step = speed_step
        # viscous drag coefficient
        self.drag_coeff = 0.5

        # setup the sensor_array_template
        self._create_car_sensors()

        # setup pygame objects and attributes: rect is always needed, image
        # only if render_mode is 'human'
        self._setup_pygame()

    def step(self, a):
        """Perform the action

        Two discrete action spaces:
            1) accelerate:  NOOP[0], UP[1], DOWN[2]
            2) steer:  NOOP[0], LEFT[1], RIGHT[2]
        """
        #  logg = logging.getLogger(f"c.{__name__}.step")
        #  logg.debug(f"Doing action {action}")
        action = [a/3, a%3]
        if action[0] == 1:
            self._accelerate("up")
        elif action[0] == 2:
            self._accelerate("down")

        if action[1] == 1:
            self._steer("left")
        elif action[1] == 2:
            self._steer("right")

        # compute delta
        pos_x_d = cos(radians(360 - self.direction)) * self.speed
        pos_y_d = sin(radians(360 - self.direction)) * self.speed

        # move the car
        self.precise_x += pos_x_d
        self.precise_y += pos_y_d
        self.pos_x = int(self.precise_x)
        self.pos_y = int(self.precise_y)
        #  logg.debug(f"Car state: x {self.pos_x} y {self.pos_y} dir {self.direction}")

        # update Sprite rect and image
        self.rect = self.rot_car_rect[self.direction]
        self.rect.center = self.pos_x, self.pos_y
        if self.render_mode == "human":
            # pick the correct image and place it
            self.image = self.rot_car_image[self.direction]

    def reset(self, pos_x, pos_y, direction=0):
        """Reset the car state
        """
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.precise_x = pos_x
        self.precise_y = pos_y

        self.direction = direction  # in degrees
        self.speed = 0

        self.rect = self.rot_car_rect[self.direction]
        self.rect.center = self.pos_x, self.pos_y
        if self.render_mode == "human":
            # pick the correct image and place it
            self.image = self.rot_car_image[self.direction]

    def _steer(self, action):
        """Steer the car
        """
        if action == "left":
            self.direction += self.dir_step
            if self.direction >= 360:
                self.direction -= 360
        elif action == "right":
            self.direction -= self.dir_step
            if self.direction < 0:
                self.direction += 360

    def _accelerate(self, action):
        """Control the speed of the car
        """
        if action == "up":
            # TODO some threshold, possibly from the drag
            self.speed += self.speed_step
        elif action == "down":
            self.speed -= self.speed_step
            # MAYBE it can go in reverse?
            if self.speed < 0:
                self.speed = 0

    def _create_car_sensors(self):
        """create the array of sensors, and rotate it for all possible directions
        """
        #  logg = getMyLogger(f"c.{__class__.__name__}._create_car_sensors")
        #  logg.debug(f"Start _create_car_sensors")

        # get the template
        sensor_array_template = self._create_sensor_array_template()

        # rotate it
        self.all_sensor_array = {}
        for dire in range(0, 360, self.dir_step):
            rot_mat = compute_rot_matrix(360 - dire)
            rotated_sa = np.matmul(rot_mat, sensor_array_template)
            int_sa = np.array(rotated_sa, dtype=np.int16)
            self.all_sensor_array[dire] = int_sa.transpose()

        # reshape it
        for dire in range(0, 360, self.dir_step):
            if self.sensor_array_type == "diamond":
                self.all_sensor_array[dire] = self.all_sensor_array[dire].reshape(
                    self.viewfield_size, self.viewfield_size, 2
                )

            elif self.sensor_array_type == "lidar":
                self.all_sensor_array[dire] = self.all_sensor_array[dire].reshape(
                    self.tot_ray_num, self.ray_sensors_per_ray, 2
                )

            else:
                raise ValueError(f"Unknown sensor_array_type {self.sensor_array_type}")

    def get_current_sensor_array(self):
        """returns the translated sensor array to use
        """
        base_sa = self.all_sensor_array[self.direction]
        car_pos = np.array((self.pos_x, self.pos_y))
        translated_sa = np.add(base_sa, car_pos)
        return translated_sa

    def _create_sensor_array_template(self):
        """create the template for the sensor array

        in a convenient shape to rotate it:
        diamond:
            * has shape (2, viewfield_size^2)
        lidar:
            * has shape (2,  (ray_num*2+1) * ray_sensors_per_ray)
        """
        #  logg = getMyLogger(f"c.{__class__.__name__}._create_sensor_array_template")
        #  logg.debug(f"Start _create_sensor_array_template")

        if self.sensor_array_type == "diamond":
            if self.sensor_array_params is None:
                self.viewfield_size = 20  # number of rows/columns in the sensor
                self.viewfield_step = 10  # spacing between the dots
            else:
                self.viewfield_size = self.sensor_array_params["viewfield_size"]
                self.viewfield_step = self.sensor_array_params["viewfield_step"]

            sat = []
            for i in range(0, self.viewfield_size):
                for j in range(0, self.viewfield_size):
                    sat.append((i, j))
            sat = np.array(sat).transpose() * self.viewfield_step

            # rotate the array so that is a diamond
            rot_mat = compute_rot_matrix(-45)
            sat = np.matmul(rot_mat, sat)
            #  logg.debug(f"shape sensor_array_template {sat.shape}")

        elif self.sensor_array_type == "lidar":
            if self.sensor_array_params is None:
                self.ray_num = 7  # number of rays per side
                self.ray_step = 15  # distance between sensors along a ray
                self.ray_sensors_per_ray = 20  # number of sensors along a ray
                self.ray_max_angle = 70  # angle to sweep left/right
            else:
                self.ray_num = self.sensor_array_params["ray_num"]
                self.ray_step = self.sensor_array_params["ray_step"]
                self.ray_sensors_per_ray = self.sensor_array_params["ray_sensors_per_ray"]
                self.ray_max_angle = self.sensor_array_params["ray_max_angle"]

            self.tot_ray_num = self.ray_num * 2 + 1
            self.ray_angle = self.ray_max_angle / self.ray_num

            # create the horizontal ray
            base_ray = tuple((s, 0) for s in range(1, self.ray_sensors_per_ray + 1))
            base_ray = np.array(base_ray).transpose() * self.ray_step

            sat = []
            for r in range(-self.ray_num, self.ray_num + 1):
                rot_mat = compute_rot_matrix(r * self.ray_angle)
                rot_ray = np.matmul(rot_mat, base_ray)
                sat.append(rot_ray)

            tot_sensors = self.tot_ray_num * self.ray_sensors_per_ray
            sat = np.array(sat).transpose((1, 0, 2)).reshape(2, tot_sensors)
            #  logg.debug(f"shape sensor_array_template {sat.shape}")

        else:
            raise ValueError(f"Unknown sensor_array_type {self.sensor_array_type}")

        return sat

    def _setup_pygame(self):
        """
        """
        # generate the car image and create all rotated versions
        self._create_car_image()
        self._rotate_car_image()

        # always needed
        self.rect = self.rot_car_rect[self.direction]
        self.rect.center = self.pos_x, self.pos_y

        # undate the image attribute only if it needs to be shown
        if self.render_mode == "human":
            # pick the correct image and place it
            self.image = self.rot_car_image[self.direction]

    def _create_car_image(self):
        """create the car sprite image and the rect

        image.set_colorkey(colorkey, RLEACCEL)
        """
        #  logg = logging.getLogger(f"c.{__name__}._create_car_image")
        #  logg.info(f"Start _create_car_image")

        # wheel dimensions
        w_color = (80, 80, 80)
        w_len = 7  # horizontal length of the wheel
        w_radius = 3
        w_wid = w_radius * 2
        delta = -3  # space from car to wheel

        # car dimensions
        car_wid = 20
        car_len = 30
        # place the car so that it touches the border of the surf
        car_top = w_radius
        car_left = ceil(car_wid / 2)
        car_size = car_len + car_wid, car_wid + w_wid

        if self.render_mode == "human":
            # create a surf just big enough for the car
            car_surf = Surface(car_size)
            # convert the surface for fastest blitting
            # same pixel format as the display Surface
            car_surf = car_surf.convert()

            black = (0, 0, 0)
            car_surf.fill(black)
            # black colors will not be blit
            #  car_surf.set_colorkey(black)
            # RLEACCEL should make blitting faster
            car_surf.set_colorkey(black, pygame.RLEACCEL)
            # show the surface area to debug
            #  car_surf.fill((0, 0, 255))

            # top left wheel
            w_top = car_top - w_radius
            w_left = car_left + delta + w_radius
            self._draw_oval(car_surf, w_top, w_left, w_wid, w_len, w_color)

            # top right wheel
            w_top = car_top - w_radius
            w_left = car_left + car_len - (delta + w_radius + w_len)
            self._draw_oval(car_surf, w_top, w_left, w_wid, w_len, w_color)

            # bottom left wheel
            w_top = car_top + car_wid - w_radius
            w_left = car_left + delta + w_radius
            self._draw_oval(car_surf, w_top, w_left, w_wid, w_len, w_color)

            # bottom right wheel
            w_top = car_top + car_wid - w_radius
            w_left = car_left + car_len - (delta + w_radius + w_len)
            self._draw_oval(car_surf, w_top, w_left, w_wid, w_len, w_color)

            # body
            body_color = (255, 0, 0)
            self._draw_oval(car_surf, car_top, car_left, car_wid, car_len, body_color)

            # windshield
            wind_wid1 = 4
            wind_wid2 = 7
            # vertical mid point
            wind_mid = car_top + car_wid // 2
            # horizontal points 52 46 36 36 46
            wind_hpos = car_len - 2
            d1 = 10
            d2 = 16
            wind_points = [
                (wind_hpos + d2, wind_mid - 1),
                (wind_hpos + d1, wind_mid - wind_wid2),
                (wind_hpos, wind_mid - wind_wid1),
                (wind_hpos, wind_mid + wind_wid1 - 1),
                (wind_hpos + d1, wind_mid + wind_wid2 - 1),
                (wind_hpos + d2, wind_mid),
            ]
            wind_color = (0, 255, 255)
            pygame.draw.polygon(car_surf, wind_color, wind_points)

            self.orig_image = car_surf

        elif self.render_mode == "console":
            # if the render_mode is console, only the rect are needed (to
            # collide with the road) so an empty surface as big as the car is
            # enough, and will be only used to create the rotated rectangles
            car_surf = Surface(car_size)
            self.orig_image = car_surf

        else:
            raise ValueError(f"Unknown render mode {self.render_mode}")

    def _draw_oval(self, surf, top, left, width, length, color):
        """draw an oval on Surface surf

        horizontal circle-rect-circle 
        top left is for the rectangle
        width is the height, length is the width lol
        """
        # make width even to simplify things
        if width % 2 != 0:
            width += 1
        mid = width // 2
        w_size = mid

        w_pos = (left, top + mid)
        pygame.draw.circle(surf, color, w_pos, w_size)
        w_pos = (left + length, top + mid)
        pygame.draw.circle(surf, color, w_pos, w_size)
        # constructor Rect( left, top, width, height )
        w_rect = (left, top, length, width)
        pygame.draw.rect(surf, color, w_rect)

    def _rotate_car_image(self):
        """Create rotated copies of the surface
        """
        #  logg = logging.getLogger(f"c.{__name__}._rotate_car_image")
        #  logg.info(f"Start _rotate_car_image")
        #  if 360 % self.dir_step != 0:
        #  logg.warn(f"A dir_step that is not divisor of 360 is a bad idea")

        self.rot_car_image = {}
        self.rot_car_rect = {}
        for dire in range(0, 360, self.dir_step):
            self.rot_car_image[dire] = rotate(self.orig_image, dire)
            self.rot_car_rect[dire] = self.rot_car_image[dire].get_rect()
