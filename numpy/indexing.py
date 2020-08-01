import numpy as np

x = np.random.randint(100, size=10)

# Một số kiểu gọi biến theo chỉ mục
# print(x)
# print([x[3], x[7], x[2]])
# ind = [3, 7, 2]
# print(x[ind])
# ind = np.array([[3, 7],
#                  [2, 5]])
# print(x[ind])
#
x = np.arange(12).reshape((3, 4))
print(x)
#
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
#
# # Lấy phần tử theo hàng & cột, tương ứng chỉ mục với vector row, col
print(x[row, col])
#
# # Chuyển row thành ma trận cột, chọn các phần tử theo vị trí tương ứng
# print(x[row[:, np.newaxis], col])
# print(row[:, np.newaxis] * col)
#
# # Sorting
# Sắp xếp các giá trị
# x = np.array([2, 1, 4, 3, 5])
# print(np.sort(x))
#
# Sắp xếp, giá trị trả về là chỉ mục
# x = np.array([2, 1, 4, 3, 5])
# i = np.argsort(x)
# print(i)
# print(x[i])
#
# x = np.random.randint(0, 10, (4, 6))
# # Sắp xếp theo cột
# print(np.sort(x, axis=0))
# #
# # # Sắp xếp theo hàng
# # print(np.sort(x, axis=1))
#
# Partial Sorts: Partitioning

x_1 = np.array([8, 2, 3, 1, 6, 5, 4, 3, 2, 1, 5, 11, 33, 2])Developer 
# Nếu các bạn đã làm quen với quicksort chẳng hạn đã quen với khái niệm par(partitioning).
# Mục tiêu của biến này là chia mảng thành hai mảng sao cho mảng bên trai nhỏ hơn par và ngc lại,
# Điều này làm ta ko quan tâm là có dc sắp xếp một cách có thứ tự hay ko, chỉ cần chia đúng mảng là dc
print(np.partition(x_1, kth=4))
# print(np.partition(x, 2, axis=1))
#
#
#
