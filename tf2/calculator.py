import tensorflow as tf
from dataclasses import dataclass
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@dataclass
class Machine(object):
    def __init__(self):
        self._num1 = 0
        self._opcode = ''
        self._num2 = 0

    @property
    def num1(self) -> int: return self._num1

    @num1.setter
    def num1(self, num1): self._num1 = num1

    @property
    def opcode(self) -> str: return self._opcode

    @opcode.setter
    def opcode(self, opcode): self._opcode = opcode

    @property
    def num2(self) -> int: return self._num2

    @num2.setter
    def num2(self, num2): self._num2 = num2

'''
@tf.function 이란?
def 위에 @tf.fucnction annotation을 붙이면 마치 tf2.x 버전에서도 tf1.x 형태처럼 그래프 생성과 실행이 분리된 형태로 해당 함수내의 로직이 실행된다.
따라서 상황에 따라서 성능이 약간 향상 될 수 있다.(= 실행 속도가 약간 빨라질 수 있다.)
다만 해당 annoation을 붙이면 tf1.x처럼 해당 함수 내의 값을 바로 계산해 볼 수 없어서 디버깅이 불편해질 수 있다.
따라서 모든 로직에 대한 프로그래밍이 끝난 상태에서 @tf.fuction 을 붙이는 것이 좋다.
'''


class Solution:
    def __init__(self, payload):  # payload : 외부에서 투입되는 하이퍼파라미터 (실제값)
        self._num1 = payload._num1
        self._num2 = payload._num2

    @tf.function
    def add(self):
        return tf.add(self._num1, self._num2)

    @tf.function
    def sub(self):
        return tf.subtract(self._num1, self._num2)

    @tf.function
    def mul(self):
        return tf.multiply(self._num1, self._num2)

    @tf.function
    def div(self):
        return tf.divide(self._num1, self._num2)


class UseModel:
    def __init__(self):
        pass

    @staticmethod
    def calc(num1, opcode, num2):
        model = Machine()
        model.num1 = num1
        model.opcode = opcode
        model.num2 = num2

        solution = Solution(model)
        if opcode == '+':
            result = solution.add()
        elif opcode == '-':
            result = solution.sub()
        elif opcode == '*':
            result = solution.mul()
        else:
            result = solution.div()

        return result


if __name__ == '__main__':
    m = UseModel()
    print(m.calc(3, '+', 5))
    print(m.calc(3, '-', 5))
    print(m.calc(3, '*', 5))
    print(m.calc(3, '/', 5))
