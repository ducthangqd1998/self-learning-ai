import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from sklearn.datasets import load_iris


save_path = 'Matplotlib/fig_save'
x = np.linspace(0, 10, 100)


def simple_plot():
    # Cách tạo một hình vẽ hình sin đơn giản
    plt.plot(x, np.sin(x))
    plt.show()


def simple_line_plot():
    # Thực hiện biểu diễn các đường kẻ theo các trục của hình vẽ
    plt.style.use('seaborn-whitegrid')
    plt.plot(x, np.sin(x))
    plt.show()


def labeling_plot():
    # plt.plot(x, np.sin(x))
    # # Thiết lập tiêu đề của hình vẽ
    # plt.title("A Sine Curve")
    # plt.xlabel("x")
    # plt.ylabel("Sin(x)")
    # plt.show()

    plt.plot(x, np.sin(x), color="blue", label="sin(x)")
    plt.plot(x, np.cos(x), color="red", label="cos(x)")
    plt.axis("equal")
    plt.legend()
    plt.show()


def show_trigonometric():
    # fig = plt.figure()
    # plt.plot(x, np.sin(x), '-')
    # plt.plot(x, np.cos(x), '--')
    # fig.savefig(join(save_path, 'fig1.png'))
    # plt.show()

    plt.plot(x, np.sin(x - 0), color='blue')  # chọn màu bằng tên
    plt.plot(x, np.sin(x - 1), color='g')  # chế độ short color
    plt.plot(x, np.sin(x - 2), color='0.75')  # Thang màu xám (0-1)
    plt.plot(x, np.sin(x - 3), color='#FFDD44')  # Mã màu hex (RRGGBB từ 00 đến FF)
    plt.plot(x, np.sin(x - 4), color=(1.0, 0.2, 0.3))  # RGB tuple (0 -> 1)
    plt.plot(x, np.sin(x - 5), color='chartreuse');  # tên màu theo thuộc tính cuaur HTML
    plt.show()


def mul_interfaces():
    plt.figure()
    plt.subplot(2, 1, 1)  # [rows, columns, panel number] => panel thứ nhất
    plt.plot(x, np.sin(x))
    plt.subplot(2, 1, 2)  # panel thứ hai
    plt.plot(x, np.cos(x))
    plt.show()

    # Cách tiếp cận hướng đối tượng: Cách này tường minh và dễ dàng thực hiện hơn khi
    # chúng ta không cần quan tâm quá nhiều đến các khái niệm axis, ...
    # fig, ax = plt.subplots(2)
    # ax[0].plot(x, np.sin(x))
    # ax[1].plot(x, np.cos(x))
    # plt.show()


def scatter_plots():
    # Thực hiện biểu diễn các điểm rời rạc, color tùy các bạn
    # y = np.sin(x)
    # plt.plot(x, y, 'o', color='black')
    # plt.show()

    # marker là một tham số dùng để đánh dấu các điểm dữ liệu, có thể tham khảo các kiểu marker thông dụng
    # Tài liệu tham khảo cho các bạn https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    # markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
    # for marker in markers:
    #     plt.plot(np.random.rand(5),
    #              np.random.rand(5),
    #              marker,
    #              label="marker={}".format(marker))
    #
    # # Dùng để vẽ các legends, numpoints dùng để chú thích các marker, set bằng 1
    # plt.legend(numpoints=1)
    # # Đặt giới hạn của các trục hiện tại
    # plt.xlim(0, 2)
    # plt.show()

    x = np.linspace(0, 10, 30)
    y = np.sin(x)
    plt.plot(x, y, "-ok") # line (-), circle marker (o), black (k)
    plt.show()


def scatter_plots_2():
    # Cách thứ 2 có thể dùng hàm scatters, nó khá giống hàm plot
    # Khác nhau ở chỗ là có thể hiển thị được kích thước, màu và hình dạng riêng của từng điểm
    # x = np.linspace(0, 10, 30)
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
    # simple_plot()
    # simple_line_plot()
    # labeling_plot()
    # show_trigonometric()
    # mul_interfaces()
    # scatter_plots()
    scatter_plots_2()

if __name__ == '__main__':
    main()

