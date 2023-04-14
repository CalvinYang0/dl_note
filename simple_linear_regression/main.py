import numpy as np


def compute_loss_for_line_given_points(b, w, points):
    """
    计算损失
    """
    # 累加损失
    total_loss = 0
    # 累加所有点的损失，计算方式为(y - (w * x + b)) ** 2，即y的真实值和预测值的差的平方
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_loss += (y - (w * x + b)) ** 2
    # 返回平均损失
    return total_loss / float(len(points))


# 梯度下降函数
def gradient_descent(b, w, points, learning_rate):
    """
    梯度下降函数
    :param b:
    :param w:
    :param points:
    :param learning_rate:
    :return:
    """
    # 初始化梯度
    b_gradient = 0
    w_gradient = 0
    # 根据所有数据点计算梯度
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 计算b的梯度,2(y - (w * x + b)
        b_gradient += -2 * (y - ((w * x) + b))
        # 计算w的梯度，2x(y - (w * x + b))
        w_gradient += -2 * x * (y - ((w * x) + b))
    #平均化
    b_gradient = b_gradient / float(len(points))
    w_gradient = w_gradient / float(len(points))
    # 更新b和w
    new_b = b - (learning_rate * b_gradient)
    new_w = w - (learning_rate * w_gradient)
    return [new_b, new_w]


def gradient_descent_runner(b, w, points, learning_rate, epoch):
    """
    梯度下降迭代函数
    :param b:
    :param w:
    :param points:
    :param learning_rate:
    :param epoch:
    :return:
    """
    # 迭代epoch次
    for i in range(epoch):
        # 更新b和w
        b, w = gradient_descent(b, w, points, learning_rate)
        # 每迭代100次，打印一次误差
        if i % (epoch / 10) == 0:
            print("迭代次数：", i, "误差：", compute_loss_for_line_given_points(b, w, points))
    print("迭代次数：", epoch, "误差：", compute_loss_for_line_given_points(b, w, points))
    return [b, w]


def main():
    # 数据为data.csv内一串点储存x和y，目标是进行线性回归，形如y = wx + b，需要学习w和b
    # 使用numpy的”genfromtxt函数读取csv文件，分隔符号为逗号
    points = np.genfromtxt("data.csv", delimiter=",")
    # 设置学习率
    learning_rate = 0.0001
    # 设置初始b和w
    init_b = 0
    init_w = 0
    # 设置迭代次数
    epoch = 1000
    # 打印初始误差
    print("初始误差为：", compute_loss_for_line_given_points(init_b, init_w, points))
    # 迭代
    [b, w] = gradient_descent_runner(init_b, init_w, points, learning_rate, epoch)
    print("最终误差为：", compute_loss_for_line_given_points(b, w, points))
    print("最终w为：", w)
    print("最终b为：", b)
if __name__ == '__main__':
    main()

