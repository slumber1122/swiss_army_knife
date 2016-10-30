import math
import random

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(i, j, fill=0.0):
    m = []
    for i in range(i):
        m.append([fill] * j)
    return m


def randomize_matrix(matrix, a, b):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = random.uniform(a, b)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def dsigmoid(y):
    '''
    sigmoid 的导数
    '''
    return y * (1 - y)


class TipsException(Exception):
    pass


class NN:
    def __init__(self, ni, nh, no):
        self.ni = ni + 1
        self.nh = nh
        self.no = no

        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        self.wi = make_matrix(self.ni, self.nh)
        self.wo = make_matrix(self.nh, self.no)

        randomize_matrix(self.wi, -0.2, 0.2)
        randomize_matrix(self.wo, -2.0, 2.0)

        self.ci = make_matrix(self.ni, self.nh)
        self.co = make_matrix(self.nh, self.no)

    def run_NN(self, inputs):
        '''
        向前传播
        :param inputs: input
        :return: 类别
        '''

        if len(inputs) != self.ni - 1:
            raise TipsException('incorrect number of inputs')

        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.nh):
                sum += (self.ai[i] * self.wi[i][j])
            self.ah[j] = sigmoid(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += (self.ah[j] * self.wo[j][k])
            self.ao[k] = sigmoid(sum)
        return self.ao

    def back_propagate(self, targets, N, M):
        '''
        向后传播
        :param targets: 实例的类别
        :param N: 本次学习效率
        :param M: 上次学习效率
        '''

        # 计算输出层 deltas
        # dE/dw[j][k] = (t[k] -ao[k]) * s'(SUM(w[j][k]*ah[j])) * ah[j]
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error * dsigmoid(self.ao[k])

        # 更新输出层权值
        for j in range(self.nh):
            for k in range(self.no):
                # output_deltas[k] * self.ah[j] 才是dError/dweight[j][k]
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += N * change + M * self.co[j][k]
                self.co[j][k] = change

        # 计算隐藏层 deltas
        hiddern_deltas = [0.0] * self.nh

        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hiddern_deltas[j] = error * dsigmoid(self.ah[j])

        # 更新输入层权值
        for i in range(self.ni):
            for j in range(self.nh):
                change = hiddern_deltas[j] * self.ai[i]
                self.wi[i][j] += N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        error = 0.0
        for k in range(len(targets)):
            error = 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def train(self, patterns, max_iterations=1000, N=0.5, M=0.1):
        '''
        train nn model
        :param patterns: 训练集合
        :param max_iterations: 迭代次数
        :param N: 本次学习效率
        :param M: 上次学习效率
        '''
        for i in range(max_iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.run_NN(inputs)
                error = self.back_propagate(targets, N, M)
            if i % 50 == 0:
                print('Combined error', error)
        self.test(patterns)

    def test(self, patterns):
        '''
        test
        :param patterns: data
        '''

        for p in patterns:
            inputs = p[0]
            print('Inputs:', p[0], '--->', self.run_NN(inputs), '\tTarget',
                  p[1])


def main():
    pat = [
        [[0, 0], [1]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]
    myNN = NN(2, 2, 1)
    myNN.train(pat)

if __name__ == "__main__":
    main()
