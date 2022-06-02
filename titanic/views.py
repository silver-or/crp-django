from context.models import Model
from context.domains import Dataset


class TitanicView:
    model = Model()
    dataset = Dataset()

    def modeling(self, train, test):  # 모델 구축
        model = self.model

    def preprocess(self, train, test) -> object:
        pass
