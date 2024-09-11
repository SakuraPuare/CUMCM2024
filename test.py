import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots()


def xy_to_r_theta(position):
    x, y = position
    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return np.array([r, theta])


def reverse_spiral(theta):
    return -b * theta


def is_on_reverse_spiral(position):
    x, y = position
    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return abs(r - reverse_spiral(theta)) < 1e-2


def find_nearest_point(position, func):
    polar = xy_to_r_theta(position)
    while polar[1] < 0:
        polar[1] += 2 * np.pi
    polar[0] = func(polar[1])
    while polar[0] < xy_to_r_theta(position)[0]:
        polar[0] = func(polar[1])
        polar[1] -= 2 * np.pi
    r0, r1 = abs(polar[0] - xy_to_r_theta(position)[0]
                 ), abs(func(polar[1] + 2 * np.pi) - xy_to_r_theta(position)[0])
    min_ = min(r0, r1)
    return np.array([min_, polar[1] + 2 * np.pi if r1 < r0 else polar[1]])


ax.set_aspect('equal')

pitch = 170
b = pitch / (2 * np.pi)
# np.array([336.93289147, 279.98875465])
point = np.array([336.93289147, 279.98875465])
ax.scatter(*point, c='b', s=10)
# ax.scatter(*find_nearest_point(point, reverse_spiral), c='r', s=10)
print(is_on_reverse_spiral(point))
theta = np.linspace(16, 2 * 8 * np.pi, 1000)
r = reverse_spiral(theta)
x = r * np.cos(theta)
y = r * np.sin(theta)
ax.plot(x, y, c='r')

plt.show()
