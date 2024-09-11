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

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
color_idx = -1


def median_filter(data: np.array, kernel_size=3):
    return medfilt(data, kernel_size)


def moving_average(data: np.array, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def get_color():
    global color_idx
    color_idx += 1
    return colors[color_idx % len(colors)]


fig, ax = plt.subplots()


class Bench:
    def __init__(self, front_xy: np.array, back_xy: np.array) -> None:
        self.front_xy: np.array = front_xy
        self.back_xy: np.array = back_xy
        self.front_history: list[np.array] = [self.front_xy]
        self.back_history: list[np.array] = [self.back_xy]
        self.front_velocity: list[np.array] = []
        self.back_velocity: list[np.array] = []

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


class Simulate:
    def __init__(self, pitch: float = 55, num_turns: float = 16, v: float = 100, circle_radius: float = 16,
                 delta_time: float = 1, max_time: float = 500) -> None:
        # Parameters
        self.pitch = pitch
        self.b = self.pitch / (2 * np.pi)
        self.theta_max = 2 * np.pi * num_turns
        self.rho_max = self.spiral(self.b, self.theta_max)
        self.circle_radius = circle_radius
        
        self.benches: list[Bench] = []
        self.delta_time = delta_time



class Task:
    def __init__(self, pitch=55) -> None:
        self.benches: list[Bench] = []
        self.delta_time = 0
        self.max_time = 300
        self.circle_radius = 0

        self.num_turns = 16
        self.v = 100

        self._pitch = pitch
        self.b = self.pitch / (2 * np.pi)
        self.theta_max = 2 * np.pi * self.num_turns
        self.rho_max = self.spiral(self.b, self.theta_max)

        delta_view = 50
        self.plt_views = (-self.rho_max - delta_view,
                          self.rho_max + delta_view)
        self.views = np.array([self.rho_max] * 2)

        self.init()

    def refresh(self):
        self.b = self.pitch / (2 * np.pi)
        self.rho_max = self.spiral(self.b, self.theta_max)

    @property
    def pitch(self):
        # print('get pitch at', self._pitch)
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        # print('set pitch to', value)
        self._pitch = value
        self.refresh()

    def init(self):
        point_a_xy = self.r_theta_to_xy(self.rho_max, self.theta_max)

        # 龙头的位置
        self.benches.append(Bench(point_a_xy, point_a_xy +
                                  np.array([0, DRAGON_BODY_DISTANCE])))

        # 龙身和龙尾的位置
        for i in range(1, DRAGON_LENGTH):
            self.benches.append(
                Bench(self.benches[i - 1].back_xy, self.benches[i - 1].back_xy + np.array([0, DRAGON_BODY_DISTANCE])))

    @staticmethod
    def r_theta_to_xy(r: float, theta: float) -> np.array:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.array([x, y])

    def xy_to_r_theta(self, x: float, y: float) -> np.array:
        r = np.linalg.norm(np.array([x, y]))
        theta = r / self.b
        return np.array([r, theta])

    @staticmethod
    def spiral(b, theta):
        return b * theta

    def spiral_inv(self, r):
        return r / self.b

    def theta_to_omega(self, theta):
        return -self.v / (self.b * np.sqrt(theta ** 2 + 1))

    def draw_circle(self):
        ax.add_patch(plt.Circle((0, 0), self.circle_radius,
                                fill=True, facecolor='pink', alpha=0.5))
        ax.plot([0], [0], 'o', color='purple')

    def draw_spiral(self):
        ax.set_aspect('equal')
        ax.set_xlim(self.plt_views)
        ax.set_ylim(self.plt_views)

        theta_ = np.linspace(0, self.theta_max, 1500)
        r = self.spiral(self.b, theta_)
        x = r * np.cos(theta_)
        y = r * np.sin(theta_)
        ax.plot(x, y, 'r')

    def draw_benches(self, with_projection=True):
        for bench in self.benches:
            if (bench.front_xy > self.views).any() or (bench.back_xy > self.views).any():
                break

            if with_projection:
                projection = bench.get_projection()
                polygon = plt.Polygon(
                    projection, closed=True, fill=False, edgecolor='g')
                ax.add_patch(polygon)

            # plot point
            ax.plot(*bench.front_xy, 'bo', markersize=1)
        ax.plot(*self.benches[-1].back_xy, 'bo', markersize=1)

    def check_collision(self) -> bool:
        head = Polygon(self.benches[0].get_projection())
        geo = [Polygon(self.benches[idx].get_projection())
               for idx in range(2, DRAGON_LENGTH)]

        gdf = geopandas.GeoDataFrame(geometry=geo)

        return head.intersects(gdf.union_all())

    def find_intersection_point(self, center: np.array, radius: float) -> np.array:
        """_summary_

        Args:
            center (np.array): 圆心xy坐标
            radius (float): 圆半径

        Returns:
            np.array: 极径 r 最近的交点坐标
        """
        center_r_theta = self.xy_to_r_theta(*center)
        spiral_points = []
        for theta in np.linspace(center_r_theta[1] + np.pi, center_r_theta[1] - np.pi, 500):
            r = self.spiral(self.b, theta)
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
        return spiral_points[0][1]

    def find_back_position(self, front_xy, back_xy, length=DRAGON_BODY_DISTANCE):
        if abs(self.rho_max - front_xy[0]) < 1e-2:
            return np.array([self.rho_max, front_xy[1] + length])
        elif abs(self.rho_max - back_xy[0]) < 1e-2 and self.rho_max - front_xy[0] < length:
            k = np.tan(np.arccos((self.rho_max - front_xy[0]) / length))
            b = front_xy[1] - k * front_xy[0]
            x = self.rho_max
            y = k * x + b
            if y < 0 or k < np.tan(np.deg2rad(80)):
                return self.find_intersection_point(front_xy, length)
            return [x, y]
        else:
            return self.find_intersection_point(front_xy, length)

    def is_in_turn_around_area(self):
        assert self.circle_radius > 0
        return np.linalg.norm(self.benches[0].front_xy) < self.circle_radius

    def calculate_head_position(self):
        head = self.benches[0]

        # find the next position of the head front
        r_theta = self.xy_to_r_theta(*head.front_xy)
        delta_theta = self.theta_to_omega(r_theta[1]) * self.delta_time
        r_theta[1] += delta_theta
        r_theta[0] = self.spiral(self.b, r_theta[1])
        self.benches[0].front_xy = self.r_theta_to_xy(*r_theta)

    def calculate_other_position(self):
        head = self.benches[0]
        # find the next position of the head back
        self.benches[0].back_xy = self.find_back_position(
            head.front_xy, head.back_xy, DRAGON_HEAD_DISTANCE)

        for i in range(1, DRAGON_LENGTH):
            self.benches[i].front_xy = self.benches[i - 1].back_xy
            self.benches[i].back_xy = self.find_back_position(
                self.benches[i].front_xy, self.benches[i].back_xy)

    def log_position(self):
        for i in range(DRAGON_LENGTH):
            self.benches[i].front_history.append(self.benches[i].front_xy)
            self.benches[i].back_history.append(self.benches[i].back_xy)

    def log_velocity(self):
        for i in range(DRAGON_LENGTH):
            self.benches[i].front_velocity.append(
                (np.array(self.benches[i].front_history[-1]) - np.array(self.benches[i].front_history[-2])) / self.delta_time)
            self.benches[i].back_velocity.append(
                (np.array(self.benches[i].back_history[-1]) - np.array(self.benches[i].back_history[-2])) / self.delta_time)

    def export_latex(self):
        lines = []
        for i in self.benches:
            lines.append(
                '/'.join(map(str, np.concatenate([i.front_xy, i.back_xy]).flatten().tolist())) + ',')
        with open('./out/loc.txt', 'w') as f:
            f.write('\n'.join(lines))


class BaseTask(Task):
    def __init__(self, pitch) -> None:
        super().__init__(pitch=pitch)
        # self.main()
        self.delta_time = 1
        self.max_time = 500

    def validate(self) -> bool:
        self.init()
        current_time = 0
        while current_time <= self.max_time:
            self.calculate_head_position()
            self.calculate_other_position()

            if self.check_collision():
                print('Collision! at: ', current_time)
                print('Head position: ', self.benches[0].front_xy)
                return False

            if self.is_in_turn_around_area():
                return False

            current_time += self.delta_time

        return self.is_in_turn_around_area()

    def simulate(self):
        # for index in tqdm.trange(int(self.max_time / self.delta_time)):
        current_time = 0
        while current_time <= self.max_time:
            self.calculate_head_position()
            self.calculate_other_position()
            self.log_position()
            self.log_velocity()

            if self.check_collision():
                print('Collision! at: ', current_time)
                print('Head position: ', self.benches[0].front_xy)
                break

            current_time += self.delta_time
            print(current_time)
            print(f'Head position: {self.benches[0].front_xy}')

    def export_image(self):
        # resize the figure
        fig.set_size_inches(10, 10)
        # reduce the space between the subplots
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        current_time = 0
        while current_time <= self.max_time:
            self.calculate_head_position()
            self.calculate_other_position()
            # self.calculate_velocity()

            self.draw_spiral()
            self.draw_circle()
            self.draw_benches()
            plt.savefig(f'./out/3/{current_time // self.delta_time}.png')
            ax.clear()
            ax.set_aspect('equal')
            ax.set_xlim(self.plt_views)
            ax.set_ylim(self.plt_views)

            if self.check_collision():
                print('Collision! at: ', current_time)
                print('Head position: ', self.benches[0].front_xy)
                break

            current_time += self.delta_time
            print(current_time)
            print(f'Head position: {self.benches[0].front_xy}')

    def draw_velocity_index(self, idx):
        if idx < 0:
            idx += DRAGON_LENGTH
        front_velocity = [np.linalg.norm(v)
                          for v in self.benches[idx].front_velocity]
        back_velocity = [np.linalg.norm(v)
                         for v in self.benches[idx].back_velocity]

        # plot front velocity
        # 折线图
        ax.plot(median_filter(front_velocity),
                get_color(), label=f'{idx} front velocity')
        ax.plot(median_filter(back_velocity), label=f'{idx} back velocity')

    def main(self):
        # self.simulate()
        # self.export_latex()
        self.export_image()

        # self.run()
        # self.solve()  # self.draw_spiral()

        # self.draw_benches()

        # self.draw_velocity_index(0)  # for i in range(0, DRAGON_LENGTH, 1):  #     self.draw_velocity_index(i)  # ax.legend()

        # plt.show()  # pass


class Task1(Task):
    def __init__(self) -> None:
        super().__init__()
        self.delta_time = 1
        self.max_time = 300

        self.simulate()
        self.solve()

    def simulate(self):
        # for index in tqdm.trange(int(self.max_time / self.delta_time)):
        current_time = 0
        while current_time <= self.max_time:
            self.calculate_head_position()
            self.calculate_other_position()
            self.log_position()
            self.log_velocity()

            # if self.check_collision():
            #     print('Collision! at: ', current_time)
            #     print('Head position: ', self.benches[0].front_xy)
            #     break

            current_time += self.delta_time
            print(current_time, f'Head position: {self.benches[0].front_xy}')

    def solve(self):
        head_ = ['index'] + [f'{i} s' for i in range(self.max_time + 1)]
        location = pd.DataFrame(columns=head_)
        speed = pd.DataFrame(columns=head_)

        location['index'] = ['龙头x(m)', '龙头y(m)'] + \
                            [f"第{i}节龙身{x_or_y} (m)" for i in range(1, DRAGON_LENGTH - 1) for x_or_y in ['x', 'y']] + \
                            ['龙尾x(m)', '龙尾y(m)', '龙尾（后）x(m)', '龙尾（后）y(m)', ]
        speed['index'] = ['龙头 (m/s)'] + \
                         [f"第{i}节龙身速度 (m/s)" for i in range(1, DRAGON_LENGTH - 1)] + \
                         ['龙尾  (m/s)', '龙尾（后） (m/s)', ]

        for t in range(0, self.max_time + 1):
            loc_data = []
            speed_data = []
            for i in range(DRAGON_LENGTH - 1):
                loc_data.append(self.benches[i].front_history[t])
                speed_data.append(np.linalg.norm(
                    self.benches[i].front_velocity[t]))
            speed_data.append(np.linalg.norm(
                self.benches[-1].front_velocity[t]))
            speed_data.append(np.linalg.norm(
                self.benches[-1].back_velocity[t]))
            # speed_data = median_filter(speed_data) / 100
            speed_data = np.array(speed_data) / 100
            loc_data.append(self.benches[-2].back_history[-1])
            loc_data.append(self.benches[-1].back_history[-1])
            location[head_[t + 1]] = np.array(loc_data).flatten() / 100
            speed[head_[t + 1]] = speed_data

        with pd.ExcelWriter('./result/result1.xlsx', engine='openpyxl') as writer:
            location.to_excel(writer, index=False, sheet_name='位置')
            speed.to_excel(writer, index=False, sheet_name='速度')


class Task2(Task):
    def __init__(self) -> None:
        super().__init__()
        self.delta_time = 1
        self.max_time = 500

        self.solve()
        self.write()

    def solve(self):
        # for index in tqdm.trange(int(self.max_time / self.delta_time)):
        current_time = 0
        while True:
            self.calculate_head_position()
            self.calculate_other_position()
            self.log_position()
            self.log_velocity()

            if self.check_collision():
                print('Collision! at: ', current_time)
                print('Head position: ', self.benches[0].front_xy)
                break

            current_time += self.delta_time
            if current_time >= self.max_time:
                break

            # if current_time >= 405:
            #     self.delta_time = 1
            print(current_time, f'Head position: {self.benches[0].front_xy}')
        # self.draw_spiral()
        # self.draw_benches()

        # plt.show()
        self.export_latex()

    def write(self):
        head_ = ['index', '横坐标x (m)', '纵坐标y (m)', '速度 (m/s)']
        df = pd.DataFrame(columns=head_)
        df['index'] = ['龙头'] + \
                      [f'第{i}节龙身' for i in range(1, DRAGON_LENGTH - 1)] + ['龙尾', '龙尾（后）']

        x_data = []
        y_data = []
        speed_data = []
        for i in range(DRAGON_LENGTH):
            x_data.append(self.benches[i].front_xy[0])
            y_data.append(self.benches[i].front_xy[1])
            speed_data.append(np.linalg.norm(
                self.benches[i].front_velocity[-1]))
        x_data.append(self.benches[-1].back_xy[0])
        y_data.append(self.benches[-1].back_xy[1])
        speed_data.append(np.linalg.norm(self.benches[-1].back_velocity[-1]))
        speed_data = median_filter(speed_data) / 100
        df['横坐标x (m)'] = list(map(lambda x: x / 100, x_data))
        df['纵坐标y (m)'] = list(map(lambda x: x / 100, y_data))
        df['速度 (m/s)'] = speed_data

        with pd.ExcelWriter('./result/result2.xlsx', engine='openpyxl') as writer:
            df.to_excel(writer, index=False)


class Task3(Task):
    class Simulate(Task):
        def __init__(self, pitch) -> None:
            super().__init__(pitch=pitch)

            self.delta_time = 1
            self.max_time = 500
            self.circle_radius = 900 / 2
            # self.main()

        def validate(self) -> bool:
            print(f'Start calculate pitch: {self.pitch}')
            self.init()
            current_time = 0
            while current_time <= self.max_time:
                self.calculate_head_position()
                self.calculate_other_position()

                if self.check_collision():
                    print('Collision! at: ', current_time)
                    print('Head position: ', self.benches[0].front_xy)
                    return False

                if self.is_in_turn_around_area():
                    return True

                current_time += self.delta_time
                # print(current_time)

            return self.is_in_turn_around_area()

    def __init__(self) -> None:
        super().__init__()

        self.delta_time = 1
        self.max_time = 500

        self.circle_radius = 900 / 2

        self.solve()

    def solve(self):
        pitch_upper = 60
        pitch_lower = 0

        # Pitch: 46.025390625 is the best pitch
        # binary search
        while pitch_upper - pitch_lower > 1e-2:
            self.pitch = (pitch_upper + pitch_lower) / 2

            sim = self.Simulate(self.pitch)

            if ret := sim.validate():
                pitch_upper = self.pitch
            else:
                pitch_lower = self.pitch

            print(
                f'Pitch: {self.pitch}, upper: {pitch_upper}, lower: {pitch_lower}, status: {ret}')

        print(f'Pitch: {self.pitch} is the best pitch')


class Task4(Task):
    class Simulate(Task):
        def __init__(self) -> None:
            super().__init__(pitch=170)

            self.circle_radius = 900 / 2

        def calculate_head_position(self):
            head = self.benches[0]

            # find the next position of the head front
            r_theta = self.xy_to_r_theta(*head.front_xy)
            delta_theta = self.theta_to_omega(r_theta[1]) * self.delta_time
            r_theta[1] += delta_theta
            r_theta[0] = self.spiral(self.b, r_theta[1])
            self.benches[0].front_xy = self.r_theta_to_xy(*r_theta)

        def calculate_other_position(self):
            head = self.benches[0]
            # find the next position of the head back

            self.benches[0].back_xy = self.find_back_position(
                head.front_xy, head.back_xy, DRAGON_HEAD_DISTANCE)

            for i in range(1, DRAGON_LENGTH):
                self.benches[i].front_xy = self.benches[i - 1].back_xy
                self.benches[i].back_xy = self.find_back_position(
                    self.benches[i].front_xy, self.benches[i].back_xy)

        def find_intersection_point(self, center: np.array, radius: float) -> np.array:
            """_summary_

            Args:
                center (np.array): 圆心xy坐标
                radius (float): 圆半径

            Returns:
                np.array: 极径 r 最近的交点坐标
            """

            distance = np.linalg.norm(center)
            min_k = 0
            max_k = self.num_turns * 2

            if (ratio := radius / distance) > 1:
                center_r_theta = self.xy_to_r_theta(*center)
                spiral_points = []
                for theta in np.linspace(0, 2 * np.pi, 2000):
                    for k in range(min_k, max_k + 1):
                        r = self.spiral(self.b, theta + k * 2 * np.pi)
                        if r < center_r_theta[0]:
                            continue
                        x = r * np.cos(theta)
                        y = r * np.sin(theta)
                        coordinates = np.array([x, y])
                        distance = coordinates - center
                        spiral_points.append(
                            (np.abs(np.linalg.norm(distance) - radius), coordinates))
                spiral_points.sort(key=lambda _: _[0])
                assert len(spiral_points) > 0
                return spiral_points[0][1]

            else:
                angle = np.arctan2(center[1], center[0])
                delta_angle = np.asin(ratio)

                max_r = distance + self.pitch / 2
                min_r = max(0, distance - self.pitch / 2)

                # find the upper bound of min_k
                while True:
                    r = self.spiral(self.b, angle + min_k * 2 * np.pi)
                    if r >= min_r:
                        break
                    min_k += 1
                min_k -= 1

                # find the lower bound of max_k
                while True:
                    r = self.spiral(self.b, angle + max_k * 2 * np.pi)
                    if r <= max_r:
                        break
                    max_k -= 1
                max_k += 1

                max_theta = angle + delta_angle + np.deg2rad(1)
                min_theta = angle + delta_angle - np.deg2rad(1)
                spiral_points = []
                for theta in np.linspace(min_theta, max_theta, 100):
                    for k in range(min_k, max_k + 1):
                        r = self.spiral(self.b, theta + k * 2 * np.pi)
                        x = r * np.cos(theta)
                        y = r * np.sin(theta)
                        spiral_points.append((x, y))

                spiral_points = np.array(spiral_points)
                assert spiral_points.shape[0] > 0
                return spiral_points.mean(axis=0)

        def find_back_position(self, front_xy, back_xy, length=DRAGON_BODY_DISTANCE):
            if abs(self.rho_max - front_xy[0]) < 1e-2:
                return np.array([self.rho_max, front_xy[1] + length])
            elif abs(self.rho_max - back_xy[0]) < 1e-2 and self.rho_max - front_xy[0] < length:
                k = np.tan(np.arccos((self.rho_max - front_xy[0]) / length))
                if np.isnan(k):
                    self.draw_spiral()
                    self.draw_benches()
                    plt.show()
                    pass
                b = front_xy[1] - k * front_xy[0]
                x = self.rho_max
                y = k * x + b
                if y < 0 or k < np.tan(np.deg2rad(80)):
                    return self.find_intersection_point(front_xy, length)
                return [x, y]
            else:
                return self.find_intersection_point(front_xy, length)

        def validate(self) -> bool:
            self.init()
            current_time = 0
            while current_time <= self.max_time:
                self.calculate_head_position()
                self.calculate_other_position()

                if self.check_collision():
                    print('Collision! at: ', current_time)
                    print('Head position: ', self.benches[0].front_xy)
                    return False

                if self.is_in_turn_around_area():
                    return True

                current_time += self.delta_time

            return self.is_in_turn_around_area()

    def __init__(self) -> None:
        super().__init__(pitch=170)
        self.delta_time = 1
        self.max_time = 500
        self.circle_radius = 900 / 2

        self.main()

    @staticmethod
    def spiral_reversed(b, theta):
        return -b * theta

    def draw_spiral(self, inverse=False):
        ax.set_aspect('equal')
        ax.set_xlim(self.plt_views)
        ax.set_ylim(self.plt_views)

        r = np.linspace(0, self.rho_max, 1000)
        theta = self.spiral_inv(r)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        if inverse:
            x, y = -x, -y

        ax.plot(x, y, 'r' if inverse else 'b', alpha=0.5)

    @staticmethod
    def calculate_angle(point, center):
        """Helper function to calculate the angle of the point with respect to the center"""
        delta_x = point[0] - center[0]
        delta_y = point[1] - center[1]
        return np.arctan2(delta_y, delta_x)

    def compute_arc_angles(self, point1, point2, center, clockwise):
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
        angle1, angle2 = self.compute_arc_angles(
            point1, point2, center, clockwise)

        self.draw_arc(center, angle1, angle2, radius)

    def draw_route(self, theta):
        enter_point = self.r_theta_to_xy(self.spiral(self.b, theta), theta)
        exit_point = -enter_point
        ax.plot(*enter_point, 'ro')
        ax.plot(*exit_point, 'ro')

        length = np.linalg.norm(exit_point - enter_point)
        angle = np.arctan2(
            exit_point[1] - enter_point[1], exit_point[0] - enter_point[0])

        # 1: 2 length is sum
        first, _ = length / 2, length / 2

        split_point = enter_point + \
                      [first * np.cos(angle), first * np.sin(angle)]

        ax.plot(*enter_point, 'ro')

        # draw arc
        self.draw_arc_diameter(enter_point, split_point, clockwise=True)
        self.draw_arc_diameter(split_point, exit_point)

    def main(self):
        self.draw_spiral()
        self.draw_spiral(inverse=True)
        self.draw_circle()

        theta_upper = self.spiral_inv(self.circle_radius)
        theta_lower = theta_upper - np.pi

        self.draw_route(theta_lower)
        plt.show()
        # while theta_upper - theta_lower > 1e-2:
        #     theta = (theta_upper + theta_lower) / 2

        #     if self.check_collision():
        #         theta_upper = theta
        #     else:
        #         theta_lower = theta

        plt.show()


class Task5(Task):
    def __init__(self) -> None:
        super().__init__()
        pass


if __name__ == '__main__':
    # Task1()
    Task2()
    # Task3()
    # Task4()
    # BaseTask(52.5).main()
    # BaseTask(55).main()
