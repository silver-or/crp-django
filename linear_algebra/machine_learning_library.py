import numpy as np


def numerical_derivative(f, x):
    # f : 미분하려고 하는 다변수 함수
    # x : 모든 변수를 포함하고 있는 ndarray
    delta_x = 1e-4
    # 미분한 결과를 저장할 ndarray
    derivative_x = np.zeros_like(x)

    # iterator를 이용해서 입력된 변수 x들 각각에 대해 편미분 수행
    it = np.nditer(x, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index  # iterator의 현재 index를 tuple 형태로 추출

        # 현재 칸의 값을 잠시 저장
        tmp = x[idx]

        x[idx] = tmp + delta_x
        fx_plus_delta = f(x)  # f(x + delta_x)

        x[idx] = tmp - delta_x
        fx_minus_delta = f(x)  # f(x - delta_x)

        # 중앙치분방식
        derivative_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)

        # 데이터 원상 복구
        x[idx] = tmp

        it.iternext()

    return derivative_x
