import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置参数
pitch = 170
b = pitch / (2 * np.pi)

# 定义反向螺旋函数


def reversed_spiral(theta):
    return -b * theta


def x_y_to_r_theta(position):
    x, y = position
    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return np.array([r, theta])


# 创建图形对象和轴对象
fig, ax = plt.subplots()
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)
ax.set_aspect('equal')

# 初始化空的线对象
line, = ax.plot([], [], c='r')


def xy_to_r_theta_at_reverse_spiral(position: np.array, b: float) -> np.array:
    r = np.linalg.norm(position)
    theta = r / -b
    return np.array([r, theta])


# 生成角度数组
theta = np.linspace(-xy_to_r_theta_at_reverse_spiral(
    [281.87307848, 350.51316408], b)[1], 2 * 8 * np.pi, 1000)

# 初始化函数，用于动画


def init():
    line.set_data([], [])
    return line,

# 动画更新函数


def update(frame):
    r = reversed_spiral(theta[:frame])
    x = r * np.cos(theta[:frame])
    y = r * np.sin(theta[:frame])
    line.set_data(x, y)
    return line,


# 创建动画对象
ani = FuncAnimation(fig, update, frames=len(
    theta), init_func=init, blit=True, interval=20)

# 展示动画
plt.show()
