import tensorflow as tf
from icecream import ic


class Solution(object):
    def __init__(self):
        self.mnist = tf.keras.datasets  # 엔드 투 엔드
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None

    def hook(self):
        def print_menu():
            print('0. Exit ')
            print('1. 데이터 로드')
            print('2. 모델 생성')
            print('3. 모델을 훈련하고 평가')
            print('4. 손글씨 테스트')
            return input('메뉴 선택\n')

        while 1:
            menu = print_menu()
            if menu == '0':
                break
            elif menu == '1':
                self.load_data()
            elif menu == '2':
                self.create_model()
            elif menu == '3':
                self.training_evaluation_models()
            elif menu == '4':
                self.test()

    def load_data(self):
        mnist = self.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train, self.x_test = x_train / 255.0, x_test / 255.0  # 손글씨 이미지

    def create_model(self):
        # 모델은 내부에 식을 갖는다. (activation)
        self.model = tf.keras.models.Sequential([  # Sequential : 직렬 연산, CPU  # [] : 설정값 (context)
            tf.keras.layers.Flatten(input_shape=(28, 28)),  # input layer   # 편평
            tf.keras.layers.Dense(128, activation='relu'),  # hidden layer  # 밀집    # 필터링
            tf.keras.layers.Dropout(0.2),  # hidden layer   # 가중치 제거
            tf.keras.layers.Dense(10, activation='softmax')  # output layer (softmax 사용)    # activation 은 함수 (=식)
        ])

        # compile → build (압축) → 모듈이 된다. (옵티마이저 필요)
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',  # sparse : 드문드문 ↔ dense  # categorical : nominal or ordinal
                      metrics=['accuracy'])  # 결괏값에 대한 정확도 (ratio)

    def training_evaluation_models(self):
        self.model.fit(self.x_train, self.y_train, epochs=5)  # create_model 5회전 (fit 때문에 작동, 내부적으로 loop을 돌림, train 하는 역할)
        self.model.evaluate(self.x_test, self.y_test, verbose=2)  # verbose : 결과에 대한 코멘트 (0 : 출력 안 함, 1 : 자세히, 2 : 함축적으로)

    def test(self):
        pass


if __name__ == '__main__':
    Solution().hook()
