import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from sklearn.datasets import load_iris


save_path = 'matplotlib/fig_save'
x = np.linspace(0, 10, 100)

def show_trigonometric():

    # plt.plot(x, np.sin(x))
    # plt.plot(x, np.cos(x))

    fig = plt.figure()
    plt.plot(x, np.sin(x), '-')
    plt.plot(x, np.cos(x), '--')
    fig.savefig(join(save_path, 'fig1.png'))
    plt.show()

def mul_interfaces():
    # plt.figure()
    # plt.subplot(2, 1, 1)  # [rows, columns, panel number] => panel thứ nhất
    # plt.plot(x, np.sin(x))
    # plt.subplot(2, 1, 2)  # panel thứ hai
    # plt.plot(x, np.cos(x))
    # plt.show()

    # Cách tiếp cận hướng đối tượng: Cách này tường minh và dễ dàng thực hiện hơn khi
    # chúng ta không cần quan tâm quá nhiều đến các khái niệm axis, ...
    fig, ax = plt.subplots(2)
    ax[0].plot(x, np.sin(x))
    ax[1].plot(x, np.cos(x))
    plt.show()

def scatter_plots():
    # Thực hiện biểu diễn các điểm rời rạc, color tùy các bạn
    # y = np.sin(x)
    # plt.plot(x, y, 'o', color='black')
    # plt.show()

    # marker là một tham số dùng để đánh dấu các điểm dữ liệu, có thể tham khảo các kiểu marker thông dụng
    # Tài liệu tham khảo cho các bạn https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
    for marker in markers:
        plt.plot(np.random.rand(5),
                 np.random.rand(5),
                 marker,
                 label="marker={}".format(marker))

    # Dùng để vẽ các legends, numpoints dùng để chú thích các marker, set bằng 1
    plt.legend(numpoints=1)
    # Đặt giới hạn của các trục hiện tại
    plt.xlim(0, 2)
    plt.show()

def scatter_plots_2():
    # Cách thứ 2 có thể dùng hàm scatters, nó khá giống hàm plot
    # y = np.sin(x)
    # plt.scatter(x, y, marker='o');
    # plt.show()

    # x = np.random.randn(100)
    # y = np.random.randn(100)
    # colors = np.random.rand(100)
    # sizes = 1000 * np.random.rand(100)
    #
    # plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
    #             cmap='viridis')
    # plt.colorbar();  # Hiển thị thang màu
    # plt.show()
    iris = load_iris()
    features = iris.data.T

    plt.scatter(features[0], features[1], alpha=0.2,
                s=100 * features[3], c=iris.target, cmap='viridis')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.show()

def main():
    # show_trigonometric()
    # mul_interfaces()
    # scatter_plots()
    scatter_plots_2()

if __name__ == '__main__':
    main()

