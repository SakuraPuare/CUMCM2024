
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DRAGON_LENGTH = 223
DRAGON_HEAD_LENGTH = 341
DRAGON_BODY_LENGTH = 220
DRAGON_SPACING = 27.5
DRAGON_WIDTH = 30
DRAGON_HEAD_DISTANCE = DRAGON_HEAD_LENGTH - 2 * DRAGON_SPACING
DRAGON_BODY_DISTANCE = DRAGON_BODY_LENGTH - 2 * DRAGON_SPACING


def xy_to_r_theta(position) -> np.array:
    """_summary_

    Args:
        x (float): x坐标
        y (float): y坐标

    Returns:
        np.array: 极坐标 r, theta
    """
    x, y = position
    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return np.array([r, theta])


def xy_to_r_theta_at_spiral(position) -> np.array:
    """_summary_

    Args:
        x (float): x坐标
        y (float): y坐标

    Returns:
        np.array: 极坐标 r, theta
    """
    x, y = position
    r = np.linalg.norm([x, y])
    theta = r / b
    return np.array([r, theta])


def xy_to_r_theta_at_reversed_spiral(position) -> np.array:
    """_summary_

    Args:
        x (float): x坐标
        y (float): y坐标

    Returns:
        np.array: 极坐标 r, theta
    """
    x, y = position
    r = np.linalg.norm([x, y])
    theta = -r / b
    return np.array([r, theta])


def r_theta_to_xy(polar) -> np.array:
    """_summary_

    Args:
        r (float): 极径
        theta (float): 极角

    Returns:
        np.array: xy坐标
    """
    r, theta = polar
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])


fig, ax = plt.subplots()

pitch = 170
b = pitch / (2 * np.pi)


def spiral(theta: float) -> float:
    return b * theta


def reversed_spiral(theta: float) -> float:
    return -b * theta


def spiral_inv(r: float) -> float:
    return r / b


num_turns = 16
theta_max = num_turns * 2 * np.pi
rho_max = spiral(theta_max)
plt_views = (-rho_max - 50, rho_max + 50)


def draw_spiral(inverse=False):
    ax.set_aspect('equal')
    # ax.set_xlim(plt_views)
    # ax.set_ylim(plt_views)

    r = np.linspace(0, rho_max, 1000)
    theta = spiral_inv(r)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    if inverse:
        x, y = -x, -y

    ax.plot(x, y, 'r' if inverse else 'b', alpha=0.5)


def rotate_vector(vector: np.array, angle: float):
    x, y = vector
    rotated_x = x * np.cos(angle) - y * np.sin(angle)
    rotated_y = x * np.sin(angle) + y * np.cos(angle)
    return np.array([rotated_x, rotated_y])


def is_on_spiral(b, r, theta):
    return abs(spiral(theta) - r) < 1e-2


def is_on_reversed_spiral(b, r, theta):
    return abs(reversed_spiral(theta) - r) < 1e-2


def find_spiral_intersection(center: np.array, radius: float, check_on_polar=False) -> np.array:
    # ax.add_patch(plt.Circle(center, radius, fill=False, edgecolor='purple'))
    front_polar = xy_to_r_theta(center)
    if check_on_polar and not is_on_spiral(b, front_polar):
        polar = np.array([0, front_polar[1]])
        while polar[1] < 0:
            polar[1] += 2 * np.pi
        polar[0] = spiral(b, polar[1])
        while polar[0] < front_polar[0]:
            polar[1] += 2 * np.pi
            polar[0] = spiral(b, polar[1])
        center_r_theta = polar
    else:
        center_r_theta = xy_to_r_theta_at_spiral(center, b)
    spiral_points = []
    for theta in np.linspace(center_r_theta[1] + np.pi, center_r_theta[1] - np.pi, 200):
        r = spiral(b, theta)
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


def find_spiral_intersection(center: np.array, radius: float, a: float, b: float, check_on_polar=False) -> np.array:
    # ax.add_patch(plt.Circle(center, radius, fill=False, edgecolor='purple'))
    front_polar = xy_to_r_theta(center)

    if check_on_polar and not is_on_spiral(b, front_polar):
        polar = np.array([0, front_polar[1]])
        while polar[1] < 0:
            polar[1] += 2 * np.pi
        polar[0] = spiral(b, polar[1])
        while polar[0] < front_polar[0]:
            polar[1] += 2 * np.pi
            polar[0] = spiral(b, polar[1])
        center_r_theta = polar
    else:
        center_r_theta = xy_to_r_theta_at_spiral(center, b)

    spiral_points = []
    for theta in np.linspace(center_r_theta[1] + np.pi, center_r_theta[1] - np.pi, 200):
        r = spiral(b, theta)
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


def find_circle_intersection(front_xy, r0, center, r1):
    # Calculate the distance between the two circle centers
    d = np.linalg.norm(front_xy - center)

    # Check if there are no intersections or infinite intersections
    if d > r0 + r1:
        return []
    if d < abs(r0 - r1):
        return []
    if d == 0 and r0 == r1:
        return []

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

    ax.add_patch(plt.Circle(front_xy, r0, fill=False, edgecolor='purple'))
    ax.add_patch(plt.Circle(center, r1, fill=False, edgecolor='yellow'))

    front_polar = xy_to_r_theta(front_xy)
    a_polar = xy_to_r_theta(rotate_vector(
        a_xy - center, -enter_theta))
    b_polar = xy_to_r_theta(rotate_vector(
        b_xy - center, -enter_theta))

    front_polar[1] %= 2 * np.pi
    a_polar[1] %= 2 * np.pi
    b_polar[1] %= 2 * np.pi

    a_polar[1] += -front_polar[1]
    b_polar[1] += -front_polar[1]

    if a_polar[1] > 0:
        return a_xy
    return b_xy


enter_theta = 16.63196110724008

draw_spiral()
draw_spiral(inverse=True)
ax.add_artist(plt.Circle([0, 0], 450, color='pink', fill=True))

# turning_center = np.array([-135.59279319, -179.55387614])
# enter_point = np.array([-271.18558637, -359.10775228])
# turning_radius = np.linalg.norm(turning_center - enter_point)
# exit_point = enter_point + (turning_center - enter_point) * 4
# ax.plot(*enter_point, 'bo')
# ax.plot(*exit_point, 'ro')

# turning_center_2 = enter_point + (turning_center - enter_point) * 3
# ax.add_artist(plt.Circle(turning_center,
#               turning_radius, color='b', fill=False))
# ax.add_artist(plt.Circle(turning_center_2,
#               turning_radius, color='r', fill=False))

# calc_point = np.array([ 38.72158927, -321.819459  ])
# ax.plot(*calc_point, 'go')

# ret = find_circle_intersection(
#     calc_point, DRAGON_BODY_DISTANCE, np.array([135.59279319, 179.55387614]), turning_radius)
# # ax.plot(*ret, 'yo')

# plt.show()

# (array([-426.21034353, -445.94109699]), 165.0)
# (array([-360.45413561, -187.4517672 ]), 286.0)
# (array([336.93289147, 279.98875465]), 165.0)
calc_point = np.array([336.93289147, 279.98875465])
# calc_point = r_theta_to_xy(np.array([616.86199341,  22.79916596]))
ax.plot(*calc_point, 'go')

ret = find_reversed_spiral_intersection(calc_point, 165, check_on_polar=True)
# ax.plot(*ret, 'yo')

plt.show()
