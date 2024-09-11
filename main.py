import geopandas
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import medfilt
from shapely import Polygon

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DRAGON_LENGTH = 223
DRAGON_HEAD_LENGTH = 341
DRAGON_BODY_LENGTH = 220
DRAGON_SPACING = 27.5
DRAGON_WIDTH = 30
DRAGON_HEAD_DISTANCE = DRAGON_HEAD_LENGTH - 2 * DRAGON_SPACING
DRAGON_BODY_DISTANCE = DRAGON_BODY_LENGTH - 2 * DRAGON_SPACING


def median_filter(data: np.array, kernel_size=3):
    return medfilt(data, kernel_size)


def moving_average(data: np.array, kernel_size=2):
    return np.convolve(data, np.ones(kernel_size) / kernel_size, mode='same')


class Bench:
    def __init__(self, front_xy: np.array, back_xy: np.array) -> None:
        self._front_xy: np.array = front_xy
        self._back_xy: np.array = back_xy

        self.last_front_xy: np.array = front_xy
        self.last_back_xy: np.array = back_xy

        self.front_speed: np.array = np.array([0, 0])
        self.back_speed: np.array = np.array([0, 0])

    @property
    def front_xy(self) -> np.array:
        return self._front_xy

    @front_xy.setter
    def front_xy(self, value: np.array):
        self.last_front_xy = self._front_xy
        self._front_xy = value

    @property
    def back_xy(self) -> np.array:
        return self._back_xy

    @back_xy.setter
    def back_xy(self, value: np.array):
        self.last_back_xy = self._back_xy
        self._back_xy = value

    def __repr__(self):
        return f'Bench<front: {self.front_xy}, back: {self.back_xy}>'

    def get_projection(self) -> np.array:
        distance = np.linalg.norm(self.front_xy - self.back_xy)
        center = (self.front_xy + self.back_xy) / 2
        direction = self.front_xy - self.back_xy
        angle = np.arctan2(direction[1], direction[0])

        half_width = distance / 2 + DRAGON_SPACING
        half_height = DRAGON_WIDTH / 2
        corners = []
        for dx, dy in [(-half_width, -half_height), (half_width, -half_height), (half_width, half_height),
                       (-half_width, half_height)]:
            rotated_dx = dx * np.cos(angle) - dy * np.sin(angle)
            rotated_dy = dx * np.sin(angle) + dy * np.cos(angle)
            corners.append([center[0] + rotated_dx, center[1] + rotated_dy])

        return np.array(corners)


def spiral(b, theta):
    return b * theta


def is_on_spiral_with_xy(b, position, delta=2 * np.pi / 3):
    r = np.linalg.norm(position)
    theta = np.arctan2(position[1], position[0])
    ret = abs(r - spiral(b, theta)) < delta * b
    if not ret:
        pass
    return ret


def is_on_spiral_with_polar(b, polar):
    canny = abs(xy_to_r_theta_at_spiral(
        r_theta_to_xy(polar), b)[1] - polar[1]) / np.pi / 2
    ret = abs(canny - int(canny)) < 0.5
    # ax.plot(*r_theta_to_xy(polar), 'ro' if ret else 'bo')
    return ret


def reversed_spiral(b, theta):
    return -b * theta


def is_on_reverse_spiral(b, polar):
    return abs(reversed_spiral(b, polar[1]) - polar[0]) < 1


def spiral_inv(b, r):
    return r / b


def xy_to_r_theta_at_spiral(position: np.array, b: float) -> np.array:
    r = np.linalg.norm(position)
    theta = r / b
    return np.array([r, theta])


def xy_to_r_theta_at_reverse_spiral(position: np.array, b: float) -> np.array:
    r = np.linalg.norm(position)
    theta = r / -b
    return np.array([r, theta])


def xy_to_r_theta(position: np.array) -> np.array:
    x, y = position
    r = np.linalg.norm(position)
    theta = np.arctan2(y, x)
    return np.array([r, theta])


def r_theta_to_xy(polar: np.array) -> np.array:
    r, theta = polar
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])


def theta_to_omega(theta, v: float, b: float):
    return -v / (b * np.sqrt(theta ** 2 + 1))


def circle_omega(v: float, r: float):
    return v / r


def rotate_vector(vector: np.array, angle: float):
    x, y = vector
    rotated_x = x * np.cos(angle) - y * np.sin(angle)
    rotated_y = x * np.sin(angle) + y * np.cos(angle)
    return np.array([rotated_x, rotated_y])


class Simulate:
    def __init__(self, pitch: float = 55, num_turns: float = 16, v: float = 100, circle_radius: float = 16,
                 k: float = None,
                 delta_time: float = 1, max_time: float = 300, do_log=False, do_collision_check=False,
                 do_export_image=False) -> None:
        self.current_time = 0
        self.delta_time = delta_time
        self.max_time = max_time

        # Parameters
        self.pitch = pitch
        self.b = self.pitch / (2 * np.pi)
        self.theta_max = 2 * np.pi * num_turns
        self.v = v
        self.rho_max = spiral(self.b, self.theta_max)
        self.A_polar = np.array([self.rho_max, self.theta_max])
        self.A_xy = r_theta_to_xy(self.A_polar)

        self.k = k
        self.circle_radius = circle_radius

        if self.k is not None:
            base_theta = spiral_inv(self.b, self.circle_radius)
            if k == np.inf:
                self.enter_theta = spiral_inv(self.b, self.circle_radius)
                print('Auto calculate theta: ', self.enter_theta)
            else:
                self.enter_theta = k
                while self.enter_theta < base_theta:
                    self.enter_theta += 2 * np.pi
                self.enter_theta -= 2 * np.pi
                print('Manual calculate theta: ', self.enter_theta)

            self.enter_point = r_theta_to_xy(
                [spiral(self.b, self.enter_theta), self.enter_theta])
            self.exit_point = -self.enter_point
            self.turning_radius = np.linalg.norm(
                self.exit_point - self.enter_point) / 4
            self.turning_center = self.enter_point + \
                (self.exit_point - self.enter_point) / 4
            self.turning_center_2 = self.enter_point + \
                3 * (self.exit_point - self.enter_point) / 4

            self.turning_omega = circle_omega(self.v, self.turning_radius)
            self.turning_delta_theta = self.turning_omega * self.delta_time

            self.vector_ao = np.array([self.turning_radius, 0])

            self.turning_theta = 0

        delta_view = 100
        self.plt_views = (-self.rho_max - delta_view,
                          self.rho_max + delta_view)
        self.views = np.array([self.rho_max + 10] * 2)

        self.do_log = do_log
        self.do_collision_check = do_collision_check
        self.do_turnaround_check = self.circle_radius != 0
        self.do_export_image = do_export_image

        self.need_update_interval = do_log or do_collision_check or do_export_image

        self.benches: list[Bench] = []
        self.log: list = []

        self.first_reverse = True

        self.initialize()

    def initialize(self):
        start_point = r_theta_to_xy(self.A_polar)
        print('Start point: ', start_point)

        # 龙头的位置
        self.benches.append(
            Bench(
                start_point,
                start_point + np.array([0, DRAGON_HEAD_DISTANCE])
            )
        )

        # 龙身和龙尾的位置
        for i in range(1, DRAGON_LENGTH):
            self.benches.append(
                Bench(
                    self.benches[i - 1].back_xy,
                    self.benches[i - 1].back_xy +
                    np.array([0, DRAGON_BODY_DISTANCE])
                )
            )

    def update_head_position_in_spiral(self, r_theta: np.array):
        delta_theta = theta_to_omega(
            r_theta[1], self.v, self.b) * self.delta_time
        r_theta[1] += delta_theta
        r_theta[0] = spiral(self.b, r_theta[1])
        self.benches[0].front_xy = r_theta_to_xy(r_theta)

    def update_head_position_in_reversed_spiral(self):
        r_theta = xy_to_r_theta_at_reverse_spiral(
            self.benches[0].front_xy, self.b)
        delta_theta = theta_to_omega(
            r_theta[1], self.v, self.b) * self.delta_time
        # delta_theta = 10
        r_theta[1] = -r_theta[1]
        r_theta[1] -= delta_theta
        r_theta[0] = reversed_spiral(self.b, r_theta[1])
        self.benches[0].front_xy = r_theta_to_xy(r_theta)
        # ax.plot(*self.benches[0].front_xy, 'ro')

    def update_head_position_in_turnaround(self):
        self.turning_theta += self.turning_delta_theta
        print('Turning theta: ', self.turning_theta)

        if self.turning_theta < np.pi:
            vector_ob = r_theta_to_xy(
                [self.turning_radius, np.pi - self.turning_theta])

            ds = self.vector_ao + vector_ob
            # plt.arrow(-self.turning_radius, 0, *ds, head_width=5, color='red')

            rotated_ds = rotate_vector(ds, self.enter_theta)
            current_position = self.enter_point - rotated_ds
            # plt.arrow(*self.enter_point, *-rotated_ds,
            #           head_width=5, color='blue')
        else:
            vector_ob = r_theta_to_xy(
                [self.turning_radius, self.turning_theta])

            ds = self.vector_ao + vector_ob
            # plt.arrow(-self.turning_radius, 0, *ds, head_width=5, color='red')

            rotated_ds = rotate_vector(ds, self.enter_theta)
            current_position = -rotated_ds
            # plt.arrow(0, 0, *-rotated_ds, head_width=5, color='blue')

        self.benches[0].front_xy = current_position
        # ax.plot(*current_position, 'go')

    def update_head(self):
        r_theta = xy_to_r_theta_at_spiral(self.benches[0].front_xy, self.b)
        if self.circle_radius == 0 or np.linalg.norm(self.benches[0].front_xy) > self.circle_radius and \
                self.turning_theta == 0:
            self.update_head_position_in_spiral(r_theta)
        elif self.turning_theta + self.turning_delta_theta <= 2 * np.pi:
            self.update_head_position_in_turnaround()
        # elif self.turning_theta + self.turning_delta_theta > 2 * np.pi:
        else:
            self.update_head_position_in_reversed_spiral()
            print('use')
        # else:
        #     self.update_head_position_in_spiral(r_theta)

    def update_benches(self):
        self.benches[0].back_xy = self.update_back_position(0,
                                                            self.benches[0].front_xy, DRAGON_HEAD_DISTANCE)

        for i in range(1, DRAGON_LENGTH):
            self.benches[i].front_xy = self.benches[i - 1].back_xy
            self.benches[i].back_xy = self.update_back_position(i,
                                                                self.benches[i].front_xy)

    def update_velocity(self):
        for i in range(DRAGON_LENGTH):
            self.benches[i].front_speed = np.array(
                (self.benches[i].front_xy -
                 self.benches[i].last_front_xy) / self.delta_time
            )
            self.benches[i].back_speed = np.array(
                (self.benches[i].back_xy -
                 self.benches[i].last_back_xy) / self.delta_time
            )

    def update_back_position(self, index, front_xy, length=DRAGON_BODY_DISTANCE):
        front_delta_xy = front_xy - np.array(self.A_xy)
        front_polar = xy_to_r_theta(front_xy)
        front_polar_at_spiral = xy_to_r_theta_at_spiral(front_xy, self.b)
        front_distance_to_origin = np.linalg.norm(front_xy)
        front_distance_to_A = np.linalg.norm(front_xy - self.A_xy)

        if abs(front_delta_xy[0] - self.rho_max) < self.pitch / 2:
            return np.array([self.rho_max, front_xy[1] + length])

        if front_polar[0] >= self.rho_max:
            return np.array([self.rho_max, front_xy[1] + length])

        if front_distance_to_A < length:
            if self.rho_max - np.pi / 2 < front_polar_at_spiral[1]:
                return self.find_spiral_intersection(front_xy, length)
            k = np.tan(np.arccos((self.rho_max - front_xy[0]) / length))
            b = front_xy[1] - k * front_xy[0]
            x = self.rho_max
            y = k * x + b
            if y < 0 or k < np.tan(np.deg2rad(80)):
                return self.find_spiral_intersection(front_xy, length)
            return np.array([x, y])

        front_distance_to_exit = np.linalg.norm(front_xy - self.exit_point)
        front_distance_to_enter = np.linalg.norm(
            front_xy - self.enter_point)
        if front_polar_at_spiral[0] < self.turning_radius * 2:

            front_distance_to_center = np.linalg.norm(
                front_xy - self.turning_center)
            front_distance_to_center_2 = np.linalg.norm(
                front_xy - self.turning_center_2)

            if front_distance_to_center <= self.turning_radius + 1e-2:
                if front_distance_to_enter < length:
                    ax.scatter(*front_xy, color='red', s=100)
                    return self.find_spiral_intersection(front_xy, length, check_on_polar=False)
                elif front_distance_to_exit < length:
                    ax.scatter(*front_xy, color='purple', s=100)
                    # return self.find_reversed_spiral_intersection(front_xy, length, check_on_polar=False)
                    return self.find_circle_intersection(front_xy, length, self.turning_center_2, self.turning_radius, reverse=True)
                else:
                    ax.scatter(*front_xy, color='green', s=100)
                    return self.find_circle_intersection(front_xy, length, self.turning_center, self.turning_radius)

            if front_distance_to_center_2 <= self.turning_radius + 1e-1:
                if front_distance_to_origin < length:
                    ax.scatter(*front_xy, color='blue', s=100)
                    ax.add_patch(plt.Circle(front_xy, length,
                                            fill=False, edgecolor='yellow'))
                    ret = self.find_circle_intersection(
                        front_xy, length, self.turning_center, self.turning_radius, reverse=True)
                    if xy_to_r_theta(ret - self.turning_center)[1] < 0:
                        return self.find_circle_intersection(front_xy, length, self.turning_center, self.turning_radius)
                    return ret
                else:
                    ax.scatter(*front_xy, color='brown', s=100)
                    ret = self.find_circle_intersection(
                        front_xy, length, self.turning_center_2, self.turning_radius, reverse=True)
                    if xy_to_r_theta(ret - self.turning_center_2)[1] < xy_to_r_theta(front_xy - self.turning_center_2)[1]:
                        return ret
                    return self.find_circle_intersection(front_xy, length, self.turning_center_2, self.turning_radius)
        else:
            if not front_polar[0] > self.turning_radius * 2.1 and front_distance_to_exit < length and not is_on_spiral_with_xy(self.b, front_xy):
                ax.scatter(*front_xy, color='orange', s=100)
                return self.find_circle_intersection(front_xy, length, self.turning_center_2, self.turning_radius, reverse=True)

            if index == 0 and not is_on_spiral_with_polar(self.b, front_polar):
                ax.scatter(*front_xy, color='pink', s=100)
                return self.find_reversed_spiral_intersection(front_xy, length)
            else:
                if index == 0 and not is_on_spiral_with_xy(self.b, front_xy):
                    ax.scatter(*front_xy, color='yellow', s=100)
                    return self.find_reversed_spiral_intersection(front_xy, length)

                if front_polar[1] > self.turning_radius * 2 and \
                    is_on_spiral_with_xy(self.b, self.benches[index - 1].front_xy) and \
                        front_distance_to_enter > length:
                    ax.scatter(*front_xy, color='cyan', s=100)
                    return self.find_spiral_intersection(front_xy, length)
                else:
                    if is_on_spiral_with_xy(self.b, self.benches[index - 1].front_xy):
                        ax.scatter(*front_xy, color='purple', s=100)
                        return self.find_spiral_intersection(front_xy, length)
                    return self.find_reversed_spiral_intersection(front_xy, length)

        ax.scatter(*front_xy, color='pink', s=100)
        return self.find_circle_intersection(front_xy, length, self.turning_center, self.turning_radius)

    def is_collision(self):
        head = Polygon(self.benches[0].get_projection())
        geo = [Polygon(self.benches[idx].get_projection())
               for idx in range(2, DRAGON_LENGTH)]

        gdf = geopandas.GeoDataFrame(geometry=geo)

        return head.intersects(gdf.union_all())

    def is_in_circle(self):
        return np.linalg.norm(self.benches[0].front_xy) < self.circle_radius

    def is_in_turnaround(self, position: np.array):
        return abs(np.linalg.norm(position - self.turning_center) - self.turning_radius) < 1e-2 or \
            abs(np.linalg.norm(position - self.turning_center_2) -
                self.turning_radius) < 1e-2

    def find_spiral_intersection(self, center: np.array, radius: float, check_on_polar=False) -> np.array:
        # ax.add_patch(plt.Circle(center, radius, fill=False, edgecolor='purple'))
        front_polar = xy_to_r_theta(center)
        if check_on_polar and not is_on_spiral_with_polar(self.b, front_polar):
            polar = np.array([0, front_polar[1]])
            while polar[1] < 0:
                polar[1] += 2 * np.pi
            polar[0] = spiral(self.b, polar[1])
            while polar[0] < front_polar[0]:
                polar[1] += 2 * np.pi
                polar[0] = spiral(self.b, polar[1])
            center_r_theta = polar
        else:
            center_r_theta = xy_to_r_theta_at_spiral(center, self.b)
        spiral_points = []
        for theta in np.linspace(center_r_theta[1] + np.pi, center_r_theta[1] - np.pi, 200):
            r = spiral(self.b, theta)
            if r < center_r_theta[0]:
                break
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            coordinates = np.array([x, y])
            distance = coordinates - center
            spiral_points.append(
                (np.abs(np.linalg.norm(distance) - radius), coordinates))
        spiral_points.sort(key=lambda _: _[0])
        assert len(spiral_points) > 0
        return np.array(spiral_points[0][1])

    def find_reversed_spiral_intersection(self, center: np.array, radius: float, check_on_polar=False) -> np.array:
        return -self.find_spiral_intersection(-center, radius, check_on_polar)

    def find_circle_intersection(self, front_xy, r0, center, r1, reverse=False) -> np.array:
        # ax.add_patch(plt.Circle(front_xy, r0, fill=False, edgecolor='purple'))
        # ax.add_patch(plt.Circle(center, r1, fill=False, edgecolor='yellow'))

        # Calculate the distance between the two circle centers
        d = np.linalg.norm(front_xy - center)

        # Check if there are no intersections or infinite intersections
        if d > r0 + r1 or d < abs(r0 - r1) or (d == 0 and r0 == r1):
            ax.add_patch(plt.Circle(
                front_xy, r0, fill=False, edgecolor='purple'))
            ax.add_patch(plt.Circle(
                center, r1, fill=False, edgecolor='yellow'))
            plt.show()
            raise ValueError('No intersection')

        # Calculate the distance from point 0 to point 2
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)

        # Calculate the distance from point 2 to either of the intersection points
        h = np.sqrt(r0 ** 2 - a ** 2)

        # Calculate the intersection points
        x2 = front_xy[0] + a * (center[0] - front_xy[0]) / d
        y2 = front_xy[1] + a * (center[1] - front_xy[1]) / d

        a_xy = np.array([x2 + h * (center[1] - front_xy[1]) / d,
                         y2 - h * (center[0] - front_xy[0]) / d])
        b_xy = np.array([x2 - h * (center[1] - front_xy[1]) / d,
                         y2 + h * (center[0] - front_xy[0]) / d])

        if d == r0 + r1:
            return a_xy

        front_polar = xy_to_r_theta(front_xy)
        a_polar = xy_to_r_theta(rotate_vector(
            a_xy - center, -self.enter_theta))
        b_polar = xy_to_r_theta(rotate_vector(
            b_xy - center, -self.enter_theta))

        front_polar[1] %= 2 * np.pi
        a_polar[1] %= 2 * np.pi
        b_polar[1] %= 2 * np.pi

        a_polar[1] += -front_polar[1]
        b_polar[1] += -front_polar[1]

        if reverse:
            if a_polar[1] < 0:
                return a_xy
        else:
            if a_polar[1] > 0:
                return a_xy
        return b_xy

    def draw_circle(self):
        if self.circle_radius == 0:
            return

        ax.add_patch(plt.Circle((0, 0), self.circle_radius,
                                fill=True, facecolor='pink', alpha=0.5))
        # ax.plot([0], [0], 'o', color='purple')

    def draw_spiral(self, inverse=False):
        ax.set_aspect('equal')
        # ax.set_xlim(self.plt_views)
        # ax.set_ylim(self.plt_views)

        r = np.linspace(0, self.rho_max, 2000)
        theta = spiral_inv(self.b, r)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        if inverse:
            x, y = -x, -y

        ax.plot(x, y, 'r' if inverse else 'b', alpha=0.5)

    def draw_benches(self, with_projection=True):
        # head
        for bench in self.benches:
            if (bench.front_xy > self.views).any() or (bench.back_xy > self.views).any():
                break

            if with_projection:
                projection = bench.get_projection()
                polygon = plt.Polygon(
                    projection, closed=True, fill=False, edgecolor='g', linewidth=2)
                ax.add_patch(polygon)

            # plot point
            # ax.plot(*bench.front_xy, 'bo', markersize=1)

        if (self.benches[-1].back_xy > self.views).any() or (self.benches[-1].front_xy > self.views).any():
            return

        # ax.plot(*self.benches[-1].back_xy, 'bo', markersize=1)

    def draw_route(self):
        if self.k is None:
            return
        # ax.plot(*self.enter_point, 'bo')
        # ax.plot(*self.exit_point, 'ro')
        length = np.linalg.norm(self.exit_point - self.enter_point)
        angle = np.arctan2(
            self.exit_point[1] - self.enter_point[1], self.exit_point[0] - self.enter_point[0])

        first, _ = length / 2, length / 2

        split_point = self.enter_point + \
            [first * np.cos(angle), first * np.sin(angle)]

        # draw arc
        self.draw_arc_diameter(self.enter_point, split_point, clockwise=True)
        self.draw_arc_diameter(split_point, self.exit_point)

    @staticmethod
    def draw_arc(center, angle1, angle2, a, b=None):
        """Draw arc or ellipse arc given angles and radii"""
        angles = np.linspace(angle1, angle2, 500)

        if b is None:  # Circle arc
            arc_x = center[0] + a * np.cos(angles)
            arc_y = center[1] + a * np.sin(angles)
        else:  # Ellipse arc
            arc_x = center[0] + a * np.cos(angles)
            arc_y = center[1] + b * np.sin(angles)

        ax.plot(arc_x, arc_y)

    def draw_arc_diameter(self, point1, point2, clockwise=False):
        center = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
        radius = np.linalg.norm(np.array(point1) - np.array(center))
        angle1, angle2 = self.calculate_arc_angles(
            point1, point2, center, clockwise)

        self.draw_arc(center, angle1, angle2, radius)

    @staticmethod
    def calculate_angle(point, center):
        """Helper function to calculate the angle of the point with respect to the center"""
        delta_x = point[0] - center[0]
        delta_y = point[1] - center[1]
        return np.arctan2(delta_y, delta_x)

    def calculate_arc_angles(self, point1, point2, center, clockwise):
        """Helper function to compute normalized angles and handle direction"""
        angle1 = self.calculate_angle(point1, center)
        angle2 = self.calculate_angle(point2, center)

        if angle1 < 0:
            angle1 += 2 * np.pi
        if angle2 < 0:
            angle2 += 2 * np.pi

        if clockwise:
            if angle1 < angle2:
                angle1 += 2 * np.pi
        else:
            if angle1 > angle2:
                angle2 += 2 * np.pi

        return angle1, angle2

    def export_image(self):
        self.draw_spiral()
        self.draw_circle()
        self.draw_benches()

        plt.savefig(f'./out/3/{int(self.current_time / self.delta_time)}.png')
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(self.plt_views)
        ax.set_ylim(self.plt_views)

    def export_latex(self):
        lines = []
        for i in self.benches:
            lines.append(
                '/'.join(map(str, np.concatenate([i.front_xy, i.back_xy]).flatten().tolist())) + ',')
        with open('./out/loc.txt', 'w') as f:
            f.write('\n'.join(lines))

    def run(self):
        global fig, ax
        while self.current_time < self.max_time:
            if self.do_export_image:
                fig.set_size_inches(10, 10)
                plt.subplots_adjust(left=0.01, right=0.99,
                                    top=0.99, bottom=0.01)
                self.export_image()

            self.update_head()

            if self.need_update_interval:
                self.update_benches()

            if self.do_log:
                self.update_velocity()

                if self.current_time % 1 - 1 < 1e-2:
                    self.log.append(
                        [
                            [bench.front_xy for bench in self.benches],
                            [bench.back_xy for bench in self.benches],
                            [bench.front_speed for bench in self.benches],
                            [bench.back_speed for bench in self.benches],
                        ]
                    )

            if self.do_collision_check:
                if self.is_collision():
                    print('Collision detected at ', self.current_time)
                    print('Head: ', self.benches[0].front_xy)
                    return False

            if self.do_turnaround_check and self.k is None:
                if self.is_in_circle():
                    print('Turnaround detected at ', self.current_time)
                    print('Head: ', self.benches[0].front_xy)
                    return True

            self.current_time += self.delta_time
            if self.current_time > self.max_time:
                return True

            # if self.need_update_interval:
            print('Current time: ', self.current_time,
                  'Head: ', self.benches[0].front_xy)
        return True


class Task1:
    def __init__(self):
        self.sim = Simulate(
            pitch=55,
            num_turns=16,
            v=100,
            k=np.inf,
            circle_radius=0,
            delta_time=1,
            max_time=301,
            do_log=True,
            do_collision_check=False,
            do_export_image=True
        )
        self.sim.run()
        self.write_log()
        self.sim.export_latex()

    def write_log(self):
        head_ = ['index'] + [f'{i} s' for i in range(int(self.sim.max_time))]
        location = pd.DataFrame(columns=head_)
        speed = pd.DataFrame(columns=head_)

        location['index'] = ['龙头x(m)', '龙头y(m)'] + \
                            [f"第{i}节龙身{x_or_y} (m)" for i in range(1, DRAGON_LENGTH - 1) for x_or_y in ['x', 'y']] + \
                            ['龙尾x(m)', '龙尾y(m)', '龙尾（后）x(m)', '龙尾（后）y(m)', ]
        speed['index'] = ['龙头 (m/s)'] + \
                         [f"第{i}节龙身速度 (m/s)" for i in range(1, DRAGON_LENGTH - 1)] + \
                         ['龙尾  (m/s)', '龙尾（后） (m/s)', ]

        for t in range(0, int(self.sim.max_time)):
            loc_data = []
            speed_data = []
            idx = t // self.sim.delta_time
            front_xy_data, back_xy_data, front_speed_data, back_speed_data = self.sim.log[int(
                idx)]

            for i in range(DRAGON_LENGTH - 1):
                loc_data.append(front_xy_data[i][0])
                loc_data.append(front_xy_data[i][1])
                speed_data.append(np.linalg.norm(front_speed_data[i]))
            speed_data.append(np.linalg.norm(front_speed_data[-1]))
            speed_data.append(np.linalg.norm(back_speed_data[-1]))
            loc_data.append(back_xy_data[-1][0])
            loc_data.append(back_xy_data[-1][1])
            loc_data.append(back_xy_data[-2][0])
            loc_data.append(back_xy_data[-2][1])

            location[head_[t + 1]] = np.array(loc_data).flatten() / 100
            speed[head_[t + 1]
                  ] = moving_average(speed_data, 8) / 100

        with pd.ExcelWriter('./result/result1.xlsx', engine='openpyxl') as writer:
            location.to_excel(writer, index=False, sheet_name='位置')
            speed.to_excel(writer, index=False, sheet_name='速度')


class Task2:
    def __init__(self):
        self.sim = Simulate(
            pitch=55,
            num_turns=16,
            v=100,
            k=np.inf,
            circle_radius=0,
            delta_time=1,
            max_time=500,
            do_log=True,
            do_collision_check=True,
            do_export_image=True
        )
        self.sim.run()

        fig.set_size_inches(10, 10)
        plt.subplots_adjust(left=0.01, right=0.99,
                            top=0.99, bottom=0.01)
        self.sim.draw_spiral()
        self.sim.draw_circle()
        self.sim.draw_benches()

        ax.set_aspect('equal')
        delta_view = 10
        views = self.sim.benches[0].front_xy
        ax.set_xlim(-views[0] - delta_view, views[0] + delta_view)
        ax.set_ylim(-views[1] - delta_view, views[1] + delta_view)
        plt.savefig(f'./out/3/collide.png')

        self.write_log()

    def write_log(self):
        head_ = ['index', '横坐标x (m)', '纵坐标y (m)', '速度 (m/s)']
        df = pd.DataFrame(columns=head_)
        df['index'] = ['龙头'] + \
                      [f'第{i}节龙身' for i in range(
                          1, DRAGON_LENGTH - 1)] + ['龙尾', '龙尾（后）']

        front_xy_data, back_xy_data, front_speed_data, back_speed_data = self.sim.log[-1]
        x_data = []
        y_data = []
        speed_data = []

        for i in range(DRAGON_LENGTH):
            x_data.append(front_xy_data[i][0])
            y_data.append(front_xy_data[i][1])
            speed_data.append(np.linalg.norm(front_speed_data[i]))

        x_data.append(back_xy_data[-1][0])
        y_data.append(back_xy_data[-1][1])
        speed_data.append(np.linalg.norm(back_speed_data[-1]))

        speed_data = moving_average(median_filter(speed_data)) / 100
        df['横坐标x (m)'] = list(map(lambda x: x / 100, x_data))
        df['纵坐标y (m)'] = list(map(lambda x: x / 100, y_data))
        df['速度 (m/s)'] = speed_data

        with pd.ExcelWriter('./result/result2.xlsx', engine='openpyxl') as writer:
            df.to_excel(writer, index=False)


class Task3:
    def __init__(self):
        self.solve()

    @staticmethod
    def solve():
        pitch_upper = 43
        pitch_lower = 42

        while pitch_upper - pitch_lower > 1e-6:
            pitch = (pitch_upper + pitch_lower) / 2
            sim = Simulate(
                pitch=pitch,
                num_turns=16,
                v=100,
                circle_radius=900 / 2,
                delta_time=1,
                max_time=10000,
                do_log=False,
                do_collision_check=True,
                do_export_image=False
            )
            print('Current pitch: ', pitch)
            if ret := sim.run():
                pitch_upper = pitch
            else:
                pitch_lower = pitch
            print('Result: ', ret, f'Pitch: {pitch_lower}-{pitch_upper}')
        print('Answer: ', pitch_upper)


class Task4:
    def __init__(self):
        self.sim = Simulate(
            pitch=170,
            num_turns=16,
            v=100,
            circle_radius=900 / 2,
            # k=0,
            k=np.inf,
            # k=10,
            # k=-10,
            delta_time=1,
            max_time=1300,
            # max_time=1000,
            do_log=True,
            do_collision_check=False,
            do_export_image=False
        )
        # array([-155.16098903, -431.07813439])
        # ax.plot([-155.16098903], [-431.07813439], 'o', color='pink')
        # self.sim.draw_spiral()
        # self.sim.draw_spiral(inverse=True)
        # self.sim.draw_circle()
        # self.sim.draw_route()

        self.sim.run()
        self.write_log()

        # self.sim.update_benches()
        # self.sim.export_latex()

        # self.sim.draw_benches()

        # fig.set_size_inches(10, 10)
        # plt.subplots_adjust(left=0.01, right=0.99,
        #                         top=0.99, bottom=0.01)
        # ax.set_aspect('equal')
        # ax.set_xlim(self.sim.plt_views)
        # ax.set_ylim(self.sim.plt_views)
        # plt.savefig(f'./out/turnaround.png')
        # plt.show()
        # pass
    def write_log(self):
        head_ = ['index'] + [f'{i} s' for i in range(int(self.sim.max_time))]
        location = pd.DataFrame(columns=head_)
        speed = pd.DataFrame(columns=head_)

        location['index'] = ['龙头x(m)', '龙头y(m)'] + \
                            [f"第{i}节龙身{x_or_y} (m)" for i in range(1, DRAGON_LENGTH - 1) for x_or_y in ['x', 'y']] + \
                            ['龙尾x(m)', '龙尾y(m)', '龙尾（后）x(m)', '龙尾（后）y(m)', ]
        speed['index'] = ['龙头 (m/s)'] + \
                         [f"第{i}节龙身速度 (m/s)" for i in range(1, DRAGON_LENGTH - 1)] + \
                         ['龙尾  (m/s)', '龙尾（后） (m/s)', ]

        for t in range(0, int(self.sim.max_time)):
            loc_data = []
            speed_data = []
            idx = t // self.sim.delta_time
            front_xy_data, back_xy_data, front_speed_data, back_speed_data = self.sim.log[int(
                idx)]

            for i in range(DRAGON_LENGTH - 1):
                loc_data.append(front_xy_data[i][0])
                loc_data.append(front_xy_data[i][1])
                speed_data.append(np.linalg.norm(front_speed_data[i]))
            speed_data.append(np.linalg.norm(front_speed_data[-1]))
            speed_data.append(np.linalg.norm(back_speed_data[-1]))
            loc_data.append(back_xy_data[-1][0])
            loc_data.append(back_xy_data[-1][1])
            loc_data.append(back_xy_data[-2][0])
            loc_data.append(back_xy_data[-2][1])

            location[head_[t + 1]] = np.array(loc_data).flatten() / 100
            speed[head_[t + 1]
                  ] = moving_average(speed_data, 8) / 100

        with pd.ExcelWriter('./result/result4.xlsx', engine='openpyxl') as writer:
            location.to_excel(writer, index=False, sheet_name='位置')
            speed.to_excel(writer, index=False, sheet_name='速度')


class Task5:
    def __init__(self):
        pass


fig, ax = plt.subplots()

if __name__ == '__main__':
    # Task1()
    # Task2()
    # Task3()
    Task4()
