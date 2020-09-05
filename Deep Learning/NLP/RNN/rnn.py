import numpy as np
from utils import *

def rnn_cell(xt, a_prev, parameters):
    """
	x_t: Là dữ liệu đâu vào tại thời điểm thứ t, shape của nó là (n_x, m)
	a_prev: Là hidden state tại thời điểm thứ t-1, shape của nó là (n_a, m)
	parameters: Là 1 dict chứa các tham số sau:
			- Wax: Trọng số tương ứng với dữ liệu đầu vào, shape của nó là (n_a, n_x)
			- Waa: Trọng số tương ứng với hidden state, shape của nó là (n_a, n_a)
			- Wya: Trọng số tương ứng với hidden state đầu ra, shape của nó là (n_y, n_a)
			- ba: Bias, shape của nó là (n_a, 1)
			- by: Bias tương ứng với hidden state đầu ra, shape của nó là (n_y, 1)
	"""
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    a_cur = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    y_pred = softmax(np.dot(Wya, a_cur) + by)

    cache = (a_cur, a_prev, xt, parameters)
    return a_cur, y_pred, cache 

def rnn_forward(x, a0, parameters):
    """
    Xây dưng quá trình lan truyền thuận của RNN
    Arguments:
    x -- Dữ liệu đầu vào tại mọi bước shape (n_x, m, T_x).
    a0 -- Khởi tạo hidden state, shape (n_a, m)
    parameters -- dict python chứa:
                        Waa -- Ma trận trọng số hidden state, numpy array shape (n_a, n_a)
                        Wax -- Ma trận trọng số dữ liệu đầu vào, numpy array shape (n_a, n_x)
                        Wya -- Ma trận trọng số của hidden-state tương ứng với đầu r,a numpy array shape (n_y, n_a)
                        ba --  Bias numpy array,  shape (n_a, 1)
                        by -- Bias c ủahidden-state đầu ra, numpy array shape (n_y, 1)

    Dữ liệu trả về:
    a -- Hidden states tại mọi bước, numpy array shape (n_a, m, T_x)
    y_pred -- Kết quả dự đoán tại mọi bước, numpy array shape (n_y, m, T_x)
    caches -- Giá trị cần cho quá trình backward, chứa  (list of caches, x)
    """
    
    # Khởi tạo "caches" sẽ chứa danh sách tất cả các caches
    caches = []
    
    # Lấy shape của x và Wya
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    # Khởi tạo giá trị hidden state a và y predict y_pred
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    
    # Khởi tạo a_next
    a_next = a0
    
    # Lặp
    for t in range(T_x):
        # Cập nhật hidden state kế tiếp, tính toán giá trị dự đoán và cache
        a_next, yt_pred, cache = rnn_cell(x[:,:,t], a_next, parameters)
        # Lưu giá trị new "next" hidden state
        a[:,:,t] = a_next
        # Lưu kết quả dự đoán y_pred
        y_pred[:,:,t] = yt_pred
        # Lưu cache vào caches list
        caches.append(cache)

    # Lưu các giá trị nhằm tiến hành quá trình lan truyền ngược
    caches = (caches, x)
    return a, y_pred, caches