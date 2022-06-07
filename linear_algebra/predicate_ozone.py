# Simple Linear Regression
# 온도에 따른 오존량 예측

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_algebra.machine_learning_library import numerical_derivative


class PredicateOzone:
    def __init__(self):
        self.df = pd.read_csv('./data/ozone.csv')

        # 2. Data Preprocessing(데이터 전처리)
        # 필요한 column(temp, ozone)만 추출
        self.training_data = self.df[['temp', 'ozone']]

        # 결측치 제거 - dropna() 함수 이용
        self.training_data = self.training_data.dropna(how='any')

        # 3. Training Data Set
        self.x_data = self.training_data['temp'].values.reshape(-1, 1)
        self.t_data = self.training_data['ozone'].values.reshape(-1, 1)

        # 4. Simple Linear Regression 정의
        self.W = np.random.rand(1, 1)
        self.b = np.random.rand(1)

        self.learning_rate = 1e-5
        self.f = lambda x: self.loss_func(self.x_data, self.t_data)

    # 5. loss function 정의
    def loss_func(self, x, t):
        y = np.dot(x, self.W) + self.b
        return np.mean(np.power((t - y), 2))  # 최소제곱법

    # 6. 학습 종료 후 예측값 계산 함수
    def predict(self, x):
        return np.dot(x, self.W) + self.b

    def solution(self):
        x_data = self.x_data
        t_data = self.t_data
        learning_rate = self.learning_rate
        f = self.f
        for step in range(90000):
            self.W -= learning_rate * numerical_derivative(f, self.W)
            self.b -= learning_rate * numerical_derivative(f, self.b)

            if step % 9000 == 0:
                print('W : {}, b : {}, loss : {}'.format(self.W, self.b, self.loss_func(x_data, t_data)))

        # 9. 예측
        result = self.predict(62)
        print(result)  # [[34.56270003]]

        # 10. 그래프로 확인
        plt.scatter(x_data, t_data)
        plt.plot(x_data, np.dot(x_data, self.W) + self.b, color='r')
        plt.show()


if __name__ == '__main__':
    PredicateOzone().solution()
