import numpy as np
import timeit
import time
# Tìm kiếm từ khóa này để xem cách numpy xử lí nhanh ntn nhé! vectorized operations

# Các phép toán sô học (arithmetic)
x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2) # Chia lấy số nguyên
print("-x     = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2  = ", x % 2)
print(-(0.5*x + 1) ** 2)

# Phép cộng
print(np.add(x, 2))

# Giá trị tuyệt đối
x = np.array([-2, -1, 0, 1, 2])
print(abs(x))
print(np.absolute(x))
print(np.abs(x))

# Xử lí với số phức, trả về độ lớn của biểu thức
x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
print(np.abs(x))

# Hàm lượng giác (Trigonometric)
theta = np.linspace(0, np.pi, 3)

print("theta      = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))

# Hàm lượng giác nghịch đảo
x = [-1, 0, 1]
print("x         = ", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))

# Hàm mũ (Exponents)
x = [1, 2, 3]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x))
print("3^x   =", np.power(3, x))

# Hàm logarit
x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))

# Phép nhân
x = np.arange(5)
y = np.empty(5)            # Khởi tạo một mảng gồm 5 phần tử, ghi nhớ trong bộ nhớ
np.multiply(x, 10, out=y)
print(y)

# Lũy thừa phần tử đầu tiên theo phần tử thứ 2
y = np.zeros(10)
# Kích thước x2 phải bằng với kích thước của out
np.power(2, x, out=y[::2])

# Aggregates

# Tính tổng các phâng tử trong mảng
x = np.arange(1, 6)
print(np.add.reduce(x))
print(sum(x))
print(np.sum(x))

# Tính tích các phần tử trong mảng
print(np.multiply.reduce(x))

print(np.add.accumulate(x))
print(np.multiply.accumulate(x))

# Phép nhân hai ma trận
x = np.arange(1, 6)
print(np.multiply.outer(x, x))  # Phân tử thứ 2 là ma trận chuyển vị

# Min, Max
M = np.random.random((3, 4))
print(M)
print(M.min(axis=0))
print(M.max(axis=1))

# Cộng hai mảng
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
print(a + b)
print(a + 5)

M = np.ones((3, 3))
print(M + a)

a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a)
print(b)
print(a + b)

# Giá trị trung bình
x = np.random.random((10, 3))
print(x)
print(x.mean())

# Tính giá trị trung bình của các hàng
x_mean = x.mean(axis=0)
x_centered = x - x_mean
print(x_centered.mean(0))

# So sánh logic
x = np.array([1, 2, 3, 4, 5])
print(x < 3)
print(x > 3)
print(x <= 3)
print(x >= 3)
print(x != 3)
print(x == 3)
print((2 * x) == (x ** 2))

# Đếm
x = np.random.randint(10, size=(3, 4))
print(x)

# Đếm các giá trị không bằng 0
print(np.count_nonzero(x))

# Đếm các giá trị nhỏ hơn 6
print(np.count_nonzero(x < 6))

# Tính số giá trị nhỏ hơn 6
# Các bạn thử tính xem True + True bằng bao nhiêu???
print(np.sum(x < 6))

# Tính các giá trị nhỏ hơn 6 ở mỗi hàng
print(np.sum(x < 6, axis=1))

# Có tồn tại giá trị nào lớn hơn 8 không?
print(np.any(x > 8))

# Ngc lại
print(np.any(x < 0))

# Tất cả các giá trị đều nhỏ hơn 10
print(np.all(x < 10))

# Tất cả các giá trị đều nhỏ hơn 6
print(np.all(x < 6))

# Tất cả các phần tử trên mỗi hàng đều nhỏ hơn 6
print(np.all(x < 6, axis=1))

# Chọn phần tử, cái này mình ko bt diễn đạt ntn là tốt, các bạn xem ví dụ nhé
x = np.array([[5, 0, 3, 3],
            [7, 9, 3, 5],
            [2, 4, 7, 6]])

print(x < 5)
# Trả về các phần tử có giá trị True
print(x[x < 5])

# Xác định trung vị của mảng
# Các bạn tìm hiểu trung vị khác trung bình chỗ nào nhé
x = np.array([[5, 8, 1],
            [7, 9, 6]])
print(np.median(x))
print(np.mean(x))
