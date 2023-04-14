import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import torch
import matplotlib
matplotlib.use('TkAgg')

def himmelblau(x):
    # f(x,y)=(x^2+y-11)^2+(x+y^2-7)^2
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


if __name__ == "__main__":
    # 在-6到6之间以0.1为间隔生成一个数组，在-6到6之间寻找极值点
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    # 生成网格点坐标矩阵,返回两个矩阵，分别是x和y的坐标矩阵
    X, Y = np.meshgrid(x, y)
    print('x,y range:', x.shape, y.shape)
    print('Xmaps:', X)
    print('Ymaps:', Y)
    # 计算每个点的高度值
    z = himmelblau([X, Y])
    print(z.shape)
    # 绘制曲面图
    fig = plt.figure('himmelblau')
    #gca = get current axis
    ax=fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, z)
    # 设置视角
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    #初始输入
    input=torch.tensor([-4.,0.],requires_grad=True)
    optimizer=torch.optim.Adam([input],lr=1e-3)
    for step in range(20000):
        pred=himmelblau(input)
        optimizer.zero_grad()
        pred.backward()
        optimizer.step()
        if step%2000==0:
            print('step {}: x = {}, f(x) = {}'
                  .format(step, input.tolist(), pred.item()))
# %%
