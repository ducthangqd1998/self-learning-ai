import numpy as np 

def softmax(x):
    x -= np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def main():
    print(np.tanh([[1, 2],[1, 2]]))

if __name__ == '__main__':
    main()
    