import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取Excel数据
data = pd.read_excel('result/result1.xlsx',sheet_name='速度')

# 提取位置数据
speed = data.set_index('index').T

# 计算速度：每秒位置变化的差分



# 绘制速度变化图
fig, ax = plt.subplots()

speeds = [
    speed["第1节龙身速度 (m/s)"],
    speed["第51节龙身速度 (m/s)"],
    speed["第101节龙身速度 (m/s)"],
    speed["第151节龙身速度 (m/s)"],
    speed["第201节龙身速度 (m/s)"]
]

for i, speed in enumerate(speeds):
    ax.plot(speed, label=f'第{i*50+1}节龙身速度')

ax.set_xlabel('时间 (s)')
ax.set_ylabel('速度 (m/s)')
ax.set_ylim(0, 2)
ax.legend()
plt.title('龙头到龙尾各节速度变化')


# 显示图形
# plt.xticks(rotation=90)  # 旋转x轴标签以便阅读
plt.show()