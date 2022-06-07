import numpy as np
import matplotlib.pyplot as plt
from linear_algebra.machine_learning_library import numerical_derivative


class PredicateScore:
    def __init__(self):
        # 1. Training Data Set
        # 공부 시간에 따른 점수 데이터
        self.x_data = np.array([1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 14, 15, 18, 20, 25, 28, 30]).reshape(-1, 1)
        self.t_data = np.array([5, 7, 20, 31, 40, 44, 46, 49, 60, 62, 70, 80, 85, 91, 92, 97, 98]).reshape(-1, 1)

        # 2. Linear Regression Model 정의
        self.W = np.random.rand(1, 1)  # matrix
        self.b = np.random.rand(1)  # scalar

        self.learning_rate = 0.0001

        # 미분을 진행할 loss_func에 대한 lambda 함수를 정의, lambda 함수는 항상 전역으로 사용됨
        self.f = lambda x: self.loss_func(self.x_data, self.t_data)

    # 3. Loss function
    def loss_func(self, x, t):
        y = np.dot(x, self.W) + self.b
        return np.mean(np.power((t - y), 2))  # 최소 제곱법

    # 5. prediction
    # 학습 종료 후 임의의 데이터에 대한 예측값을 알아오는 함수
    def predict(self, x):
        return np.dot(x, self.W) + self.b  # Hypothesis, Linear Regression Model # y = wx + b

    def solution(self):
        x_data = self.x_data
        t_data = self.t_data
        learning_rate = self.learning_rate
        f = self.f

        for step in range(90000):
            self.W = self.W - learning_rate * numerical_derivative(f, self.W)  # W의 편미분
            self.b = self.b - learning_rate * numerical_derivative(f, self.b)  # b의 편미분

            if step % 9000 == 0:
                print('W : {}, b : {}, loss : {}'.format(self.W, self.b, self.loss_func(x_data, t_data)))

        # 8. 학습 종료 후 예측
        print(self.predict(19))  # [[77.86823633]]

        # 데이터의 분포를 scatter로 확인
        plt.scatter(x_data.ravel(), t_data.ravel())
        plt.plot(x_data.ravel(), np.dot(x_data, self.W) + self.b)  # 직선
        plt.show()


if __name__ == '__main__':
    PredicateScore().solution()
