import numpy as np
import pandas as pd
from icecream import ic
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from context.domains import Dataset
from context.models import Model


class TitanicModel:
    model = Model()
    dataset = Dataset()

    def preprocess(self, train_fname, test_fname) -> object:  # 데이터 가공 후 리턴
        this = self.dataset  # this : model = titanic
        that = self.model
        feature = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        this.train = that.new_dframe(train_fname)
        this.test = that.new_dframe(test_fname)
        this.id = this.test['PassengerId']
        this.label = this.train['Survived']
        # Entity에서 Object로 전환
        this.train = this.train.drop(feature[1], axis=1)  # 정답 제거 # axis : 축 → row : 0, col : 1
        # self.kwargs_sample(name='이순신')  # **kwargs sample, titanic 흐름과 무관
        this = self.extract_title_from_name(this)
        title_mapping = self.remove_duplicate(this)
        this = self.title_nominal(this, title_mapping)
        this = self.sex_nominal(this)
        this = self.embarked_nominal(this)
        this = self.age_ratio(this)
        this = self.fare_ratio(this)
        this = self.drop_feature(this, feature[3:11])
        # self.df_info(this)
        k_fold = self.create_k_fold()
        accuracy = self.get_accuracy(this, k_fold)
        ic(accuracy)
        return this  # final df는 정제된 상태

    def learning(self, train_fname, test_fname):
        this = self.preprocess(train_fname, test_fname)
        k_fold = self.create_k_fold()
        ic(f'사이킷런 알고리즘 정확도 : {self.get_accuracy(this, k_fold)}')
        self.submit(this)

    @staticmethod
    def submit(this):
        clf = RandomForestClassifier()
        clf.fit(this.train, this.label)
        prediction = clf.predict(this.test)
        pd.DataFrame({'PassengerId': this.id, 'Survived': prediction}).to_csv('./save/submission.csv', index=False)

    @staticmethod
    def df_info(this):  # this = self.dataset
        [print(f'{i.info()}') for i in [this.train, this.test]]
        ic(this.train.head(3))
        ic(this.test.head(3))

    @staticmethod
    def null_check(this):
        [ic(f'{i.isnull().sum()}') for i in [this.train, this.test]]

    @staticmethod
    def id_info(this):
        ic(f'id 타입 : {type(this.id)} \n')
        ic(f'id 상위 3개 : {this.id[:3]} \n')

    @staticmethod
    def drop_feature(this, *feature) -> object:
        [i.drop(j, axis=1, inplace=True) for j in feature for i in [this.train, this.test]]  # inplace : 원본 훼손 (데이터 유실)의 가능성 존재
        # [i.drop(list(feature), axis=1, inplace=True) for i in [this.train, this.test]]
        return this

    @staticmethod
    def kwargs_sample(**kwargs) -> None:
        ic(type(kwargs))  # ic| type(kwargs): <class 'dict'>
        {print(''.join(f'key : {i}, val : {j}')) for i, j in kwargs.items()}  # key : name, val : 이순신
    '''
    Categorical vs. Continuous
    Cate → nominal (이름) vs. ordinal (순서)
    Cont → interval (음수값 허용) vs. ratio (음수값 허용 X)
    '''
    @staticmethod
    def extract_title_from_name(this) -> object:
        # this = self.dataset / train, test : DF
        for these in [this.train, this.test]:  # 데이터셋에 있던 train과 test를 편집하기 위해 리스트에 옮김
            these['Title'] = these.Name.str.extract('([A-Za-z]+)\.', expand=False)  # '' 안에서 정규식 사용 → 알파벳 대소문자 반드시 존재 / . 뒤에 있는 건 버리기
        return this

    @staticmethod
    def remove_duplicate(this):
        '''
        titles = []
        for these in [this.train, this.test]:
            titles += list(set(these['Title']))
        titles = list(set(titles))
        '''
        '''
        ['Mr', 'Sir', 'Major', 'Don', 'Rev', 'Countess', 'Lady', 'Jonkheer', 'Dr',
        'Miss', 'Col', 'Ms', 'Dona', 'Mlle', 'Mme', 'Mrs', 'Master', 'Capt']
        
        Royal : ['Countess', 'Lady', 'Sir']
        Rare : ['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona', 'Mme']
        Mr : ['Mlle']
        Ms : ['Miss']
        Master
        Mrs
        '''
        title_mapping = {'Mr': 1, 'Ms': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
        return title_mapping

    @staticmethod
    def title_nominal(this, title_mapping) -> object:
        for these in [this.train, this.test]:
            these['Title'] = these['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            these['Title'] = these['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona', 'Mme'], 'Rare')
            these['Title'] = these['Title'].replace(['Mlle'], 'Mr')
            these['Title'] = these['Title'].replace(['Miss'], 'Ms')
            these['Title'] = these['Title'].fillna(0)  # 노동자 계급
            these['Title'] = these['Title'].map(title_mapping)
        return this

    @staticmethod
    def sex_nominal(this) -> object:
        gender_mapping = {'male': 0, 'female': 1}
        for these in [this.train, this.test]:
            these['Gender'] = these['Sex'].map(gender_mapping)
        return this

    @staticmethod
    def age_ratio(this) -> object:
        train = this.train
        test = this.test
        age_mapping = {'Unknown': 0, 'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4,
                       'Young Adult': 5, 'Adult': 6,  'Senior': 7}
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]  # np.inf : 특정할 수 없는 값
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
        for these in train, test:
            these['Age'] = these['Age'].fillna(-0.5)
            these['AgeGroup'] = pd.cut(these['Age'], bins=bins, labels=labels, right=False)  # [-1.0, 0.0) -1 < 0.5 <= 0 * [ : 미포함, ) : 포함 *
            these['AgeGroup'] = these['AgeGroup'].map(age_mapping)
        return this

    @staticmethod
    def fare_ratio(this) -> object:
        train = this.train
        test = this.test
        fare_mapping = {1, 2, 3, 4}
        test['Fare'] = test['Fare'].fillna(1)
        for these in train, test:
            these['FareBand'] = pd.qcut(these['Fare'], 4, labels=fare_mapping)  # (-0.001, 7.91]
        return this

    @staticmethod
    def embarked_nominal(this) -> object:
        embarked_mapping = {'S': 1, 'C': 2, 'Q': 3}
        this.train = this.train.fillna({'Embarked': 'S'})
        for these in [this.train, this.test]:
            these['Embarked'] = these['Embarked'].map(embarked_mapping)
        return this

    @staticmethod
    def create_k_fold() -> object:
        return KFold(n_splits=10, shuffle=True, random_state=0)

    @staticmethod
    def get_accuracy(this, k_fold):
        score = cross_val_score(RandomForestClassifier(), this.train, this.label, cv=k_fold, n_jobs=1, scoring='accuracy')
        return round(np.mean(score) * 100, 2)
