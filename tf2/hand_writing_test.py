# 저장된 모델 복원하기
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model

_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # 데이터 정규화
model = load_model('./save/mnist_model.h5')

# 모델 불러오기
# model.summary()

model.evaluate(x_test, y_test, verbose=2)

plt.imshow(x_test[30], cmap="gray")
plt.show()  # 9

picks = [30]
predicted = model.predict(x_test[picks], verbose=0).argmax(axis=-1)
print("손글씨 예측값 : ", predicted)
