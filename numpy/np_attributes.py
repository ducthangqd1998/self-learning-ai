import numpy as np

x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(3, 4))
x3 = np.random.randint(10, size=(3, 4, 5))

# print("x3 ndim: ", x3.ndim)   # Số chiều của mảng numpy
# print("x3 shape:", x3.shape)  # Định dạng của mảng numpy
# print("x3 size: ", x3.size)   # Số lượng phần tử của mảng numpy

# print("dtype:", x3.dtype)     # Kiểu dữ liệu của mảng
#
arr = np.random.randint(low=0, high=10, size=10)
#
# Phần tử đầu tiên của mảng numpy
# print(arr)
# print(arr[0]) # arr[0][0]
# #
# # # Phần tử cuối cùng của mảng numpy
# print(arr[-1])
#
# arr = np.random.randint(low=0, high=10, size=(3, 4))
# print(arr)
# print(arr[0, 0])
# print(arr[0])
# print(arr[2, 2])
# print(arr[2, -1])
#
# # Do mảng trong numpy yêu cầu cùng kiểu giữa các phần tử,
# # nếu thay đổi giá trị 1 phần tử sang kiểu khác sẽ bị truncated phần tử đó
#
# arr = np.array([1, 2, 3, 4, 5, 6])
# print(arr[0])
# arr[0] = 3.14
# print(arr[0])
#
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Lấy 5 phần tử đầu tiên
# print(arr[:5])
# #
# # # Lấy các phần tử sau phần tử thứ 5
# print(arr[5:])
# #
# # # Lấy một mảng con cố định
# print(arr[4:7])
#
# # Lấy các phần tử chia hết cho 2
# print(arr[2::3])
#
# # Lấy các phần tủ không chia hết cho 2, bắt đầu là 1
# print(arr[1::2])
#
# # Đảo ngược mảng numpy
# print(arr[::-1])
#
arr = np.array([[12,  5,  2,  4],
                [ 7,  6,  8,  8],
                [ 1,  6,  7,  7]])
#
# 2 hàng, 3 cột
# print(arr[:2, :3])
#
# # 3 hàng, bước nhảy là 2
# print(arr[:3, :2:2])
#
# # Đảo ngược mảng
# print(arr[::-1, ::-1])
#
# # Lấy hàng đầu tiên
# print(arr[0, :])     # ~ arr[0]
# #
# # # Lấy cột đầu tiên
# print(arr[:, 0])
#
# Sao chép mảng
# arr_copy = arr.copy()
# print(arr_copy)
#
# Thay đổi kích thước mảng
# arr = np.arange(1, 10)
# print(arr)
# arr_reshape = arr.reshape((3,3))
# print(arr_reshape)
#
# # Chuyển đổi vector thành ma trận hàng hoặc cột
arr = np.array([1, 2, 3])

# print(arr.reshape((1, 3)))
# print(arr[np.newaxis, :])
# print(arr.reshape((3, 1)))
# print(arr[:, np.newaxis])
#
# Ghép hai ma trận
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])
print(np.concatenate([[x, y]], axis=1))
#
# grid = np.array([[1, 2, 3],
#                  [4, 5, 6]])
#
# # Ghép hai ma trận với nhau vào cột, không cần đối số axis
# print(np.concatenate([grid, grid]))
#
# # Ghép hai ma trận với nhau theo hàng
# print(np.concatenate([grid, grid], axis=1))
#
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])
#
# Ghép theo chiều dọc mảng
x = np.array([1, 2, 3])
print(np.vstack([x, grid]))
#
# # Ghép theo chiều ngang mảng
# y = np.array([[99],
#               [99]])
# print(np.hstack([grid, y]))
#
# Split mảng
grid = np.arange(16).reshape((4, 4))
#
# Chiều dọc
# upper, lower = np.vsplit(grid, [2])
# print(upper)
# print(lower)

# Chiều ngang
# left, right = np.hsplit(grid, [2])
# print(left)
# print(right)