from sklearn.datasets import load_iris


class Iris:
    def __init__(self):
        self.iris = load_iris()

    def main(self, num):
        return self.iris.data[num, :]


if __name__ == '__main__':
    print(Iris().main(0))
