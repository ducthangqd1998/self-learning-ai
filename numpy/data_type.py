import numpy as np

# Interger array
int_arr = np.array([1, 2, 3, 4, 5])
print(int_arr)
# Lưu ý, các phần tử trong array phải cùng loại, nếu khác loại thì sẽ upcast để cùng loại
float_arr = np.array([3.14, 1, 2, 3, 4, 5])
print(float_arr)

# Thiết lập kiểu dữ liệu cho mảng
arr = np.array([1, 2, 3, 4, 5], dtype='float32')
print(arr)

# Khởi tạo mảng numpy với list
n_arr = np.array([range(i, i + 3) for i in [2, 4, 6]])
print(n_arr)

# Tạo mảng có giá trị là 0
zeros_arr = np.zeros(shape=(4,4))
print(zeros_arr)

# Tạo mảng có giá trị là 1
ones_arr = np.ones(shape=(4,4))
print(ones_arr)

# Tạo mảng có giá trị cố định
nums_arr = np.full(shape=(4,4), fill_value=1.12)
print(nums_arr)

# Tạo mảng là một chuỗi theo cấp số cộng
arr = np.arange(start=0, stop=20, step=2)
print(arr)

# Tạo một mảng là là các giá trị cách đều nhau trong 1 khoảng giá trị
arr = np.linspace(start=0, stop=2, num=11)
print(arr)

# Tạo mảng ngẫu nhiên
arr = np.random.random(size=(3,3))
print(arr)

# Tạo ma trận ngẫu nhiên kiểu nguyên
arr = np.random.randint(low=0, high=10, size=(3,3))
print(arr)

# Tạo ma trận đơn vị
arr = np.eye(N=3)
print(arr)