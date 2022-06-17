import urllib.request
import pandas as pd
import numpy as np
import math
import re
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from icecream import ic

# %matplotlib inline #-- matplotlib를 통해 이미지를 바로 볼 수 있도록 설정
import matplotlib.pyplot as plt

# 한글 폰트 사용 설정
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgunsl.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 그래프 마이너스 기호 표시 설정
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

from context.domains import Reader, File


class Solution(Reader):
    def __init__(self, k=0.5):
        self.movie_comments = pd.DataFrame()
        self.file = File(context='./save/')
        self.okt = Okt()
        # 나이브베이즈 설정값
        self.k = k
        self.word_probs = []

    def hook(self):
        def print_menu():
            print('0. Exit')
            print('1. 전처리 : 텍스트 마이닝 (크롤링)')
            print('2. DF 정형화')  # 1, 2를 합치면 preprocess
            print('3. 영화 댓글이 긍정인지 부정인지 ratio 값으로 판단하시오.\n'
                  '너무 좋아요. 내 인생의 최고의 명작 영화.\n'
                  '이렇게 졸린 영화는 처음이야.')
            return input('메뉴 선택 \n')

        while 1:
            menu = print_menu()
            if menu == '0':
                break
            elif menu == '1':
                self.crawling()
            elif menu == '2':
                self.preprocessing()
            elif menu == '3':
                ic(self.naive_bayes_classifier('너무 좋아요. 내 인생의 최고의 명작 영화.'))

    def preprocessing(self):  # 파일 읽기 (크롤링)
        self.stereotype()
        df = self.movie_comments
        # ic(df.head(5))
        # 코멘트가 없는 리뷰 데이터(NaN) 제거
        df = df.dropna()
        # 중복 리뷰 제거
        df = df.drop_duplicates(['comment'])
        # self.reviews_info(df)
        # 긍정, 부정 리뷰 수
        df.label.value_counts()
        top10 = self.top10_movies(df)
        avg_score = self.get_avg_score(top10)
        self.visualization(avg_score, top10)

    def crawling(self):
        file = self.file
        file.fname = 'movie_reviews.txt'
        path = self.new_file(file)
        f = open(path, 'w', encoding='UTF-8')
        # 500 페이지까지 크롤링
        for no in range(1, 501):
            url = 'https://movie.naver.com/movie/point/af/list.naver?&page=%d' % no
            html = urllib.request.urlopen(url)
            soup = BeautifulSoup(html, 'html.parser')

            reviews = soup.select('tbody > tr > td.title')
            for rev in reviews:
                title = rev.select_one('a.movie').text.strip()
                score = rev.select_one('div.list_netizen_score > em').text.strip()
                comment = rev.select_one('br').next_sibling.strip()

                # 긍정/부정 리뷰 레이블 설정
                if int(score) >= 8:
                    label = 1  # 긍정 리뷰 (8~10점)
                elif int(score) <= 4:
                    label = 0  # 부정 리뷰 (0~4점)
                else:
                    label = 2

                f.write(f'{title}\t{score}\t{comment}\t{label}\n')
        f.close()

    def stereotype(self):
        file = self.file
        file.fname = 'movie_reviews.txt'
        path = self.new_file(file)
        self.movie_comments = pd.read_csv(path, delimiter='\t', names=['title', 'score', 'comment', 'label'])

    @staticmethod
    def reviews_info(df):
        # 영화 리스트 확인
        movie_lst = df.title.unique()
        ic('전체 영화 편수 : ', len(movie_lst))
        ic(movie_lst[:10])
        # 각 영화 리뷰 수 계산
        cnt_movie = df.title.value_counts()
        ic(cnt_movie[:20])
        # 각 영화 평점 분석
        info_movie = df.groupby('title')['score'].describe()
        # info_movie.sort_values(by=['count'], axis=0, ascending=False)  # lambda로 변환
        ic((lambda a, b: df.groupby(a)[b].describe())('title', 'score').sort_values(by=['count'], axis=0, ascending=False))

    @staticmethod
    def top10_movies(df):
        top10 = df.title.value_counts().sort_values(ascending=False)[:10]
        top10_title = top10.index.tolist()
        return df[df['title'].isin(top10_title)]

    @staticmethod
    def get_avg_score(top10):
        movie_title = top10.title.unique().tolist()  # -- 영화 제목 추출
        avg_score = {}  # -- {제목 : 평균} 저장
        for t in movie_title:
            avg = top10[top10['title'] == t]['score'].mean()
            avg_score[t] = avg
        return avg_score

    def visualization(self, avg_score, top10):
        plt.figure(figsize=(10, 5))
        plt.title('영화 평균 평점 (top 10: 리뷰 수)\n', fontsize=17)
        plt.xlabel('영화 제목')
        plt.ylabel('평균 평점')
        plt.xticks(rotation=20)

        for x, y in avg_score.items():
            color = np.array_str(np.where(y == max(avg_score.values()), 'orange', 'lightgrey'))
            plt.bar(x, y, color=color)
            plt.text(x, y, '%.2f' % y,
                     horizontalalignment='center',
                     verticalalignment='bottom')

        # plt.show()
        # self.rating_distribution(avg_score, top10)
        self.circle_chart(avg_score, top10)

    @staticmethod
    def rating_distribution(avg_score, top10):
        fig, axs = plt.subplots(5, 2, figsize=(15, 25))
        axs = axs.flatten()

        for title, avg, ax in zip(avg_score.keys(), avg_score.values(), axs):
            num_reviews = len(top10[top10['title'] == title])
            x = np.arange(num_reviews)
            y = top10[top10['title'] == title]['score']
            ax.set_title('\n%s (%d명)' % (title, num_reviews), fontsize=15)
            ax.set_ylim(0, 10.5, 2)
            ax.plot(x, y, 'o')
            ax.axhline(avg, color='red', linestyle='--')  # -- 평균 점선 나타내기

        plt.show()

    @staticmethod
    def circle_chart(avg_score, top10):
        fig, axs = plt.subplots(5, 2, figsize=(15, 25))
        axs = axs.flatten()
        colors = ['pink', 'gold', 'whitesmoke']
        labels = ['1 (8~10점)', '0 (1~4점)', '2 (5~7점)']

        for title, ax in zip(avg_score.keys(), axs):
            num_reviews = len(top10[top10['title'] == title])
            values = top10[top10['title'] == title]['label'].value_counts()
            ax.set_title('\n%s (%d명)' % (title, num_reviews), fontsize=15)
            ax.pie(values,
                   autopct='%1.1f%%',
                   colors=colors,
                   shadow=True,
                   startangle=90)
            ax.axis('equal')
        plt.show()

    def naive_bayes_classifier(self, doc):  # hook
        file = self.file
        file.fname = 'movie_reviews.txt'
        path = self.new_file(file)
        training_set = self.load_corpus(path)
        word_probs = self.train(training_set)
        point = self.classify(word_probs, doc)
        return self.postprocess(point)

    @staticmethod
    def load_corpus(path):
        corpus = pd.read_table(path, names=['title', 'point', 'doc', 'label'], encoding='UTF-8')
        # ic(corpus)
        '''
        ic| corpus:                            title  ...  label
            0    마녀(魔女) Part2. The Other One  ...      1
            1    마녀(魔女) Part2. The Other One  ...      0
            2                            브로커  ...      0
            3    마녀(魔女) Part2. The Other One  ...      1
            4    마녀(魔女) Part2. The Other One  ...      2
            ..                           ...  ...    ...
            578  마녀(魔女) Part2. The Other One  ...      0
            579  마녀(魔女) Part2. The Other One  ...      0
            580  마녀(魔女) Part2. The Other One  ...      0
            581                        범죄도시2  ...      1
            582  마녀(魔女) Part2. The Other One  ...      0
            
            [583 rows x 4 columns]
        '''
        corpus.drop(labels=['title', 'label'], axis=1, inplace=True)
        # ic(corpus)
        '''
        ic| corpus:      point                                                doc
            0       10                                 아 팝콘 괜히 삿네 콜라도 남았네
            1        1  감독이 바뀐건 아닌가 의심스러웠다... 중국영화 느낌에 장풍만쏘다 끝난건지...대사...
            2        1  볼수록 산으로 가는 느낌 감독의 영화스타일을 알고 갔음에도 중구난방 스토리에 배우들...
            3       10                              내가 본 국내영화중에서 가장 재미있다.
            4        6  출중한 연기자들이 연기력을 100% 발휘하지 못한 영화, 조심스럽지만 추측건데, (...
            ..     ...                                                ...
            578      1       방금 보고 나왔는데 정말 지루하고 재미없어요 스토리가 그냥 완전 별로고진짜 노잼
            579      2                                    존 나 재 미 없 음 짱난다
            580      3                       온 갖 좋은 재료 다버무려 개밥 만들어 놓은거 같음
            581     10  스토리는 단순하지만 긴장감 넘치고 위트있어서 덜 심각하게 즐기며 봤고 모든배우들 연...
            582      2                  돈없니? 액션씬좀 많이넣어라 명색의 액션영화인데 너무지루하다
            
            [583 rows x 2 columns]
        '''
        corpus.dropna(inplace=True)
        corpus.drop_duplicates(inplace=True)
        corpus = np.array(corpus)
        # ic(corpus)
        '''
        ic| corpus: array([[10, '아 팝콘 괜히 삿네 콜라도 남았네'],
                   [1,
                    '감독이 바뀐건 아닌가 의심스러웠다... 중국영화 느낌에 장풍만쏘다 끝난건지...대사중에 18만 기억에 남고...하고싶은말이 많았냐고 하기엔 그것도 부족했다본다..3를 만들기 위한 연결고리라고 해도 너무 아쉽다ㅠㅠ'],
                   [1, '볼수록 산으로 가는 느낌 감독의 영화스타일을 알고 갔음에도 중구난방 스토리에 배우들 연기력만 아까웠음'],
                   ...,
                   [3, '온 갖 좋은 재료 다버무려 개밥 만들어 놓은거 같음'],
                   [10,
                    '스토리는 단순하지만 긴장감 넘치고 위트있어서 덜 심각하게 즐기며 봤고 모든배우들 연기가  좋았어요 특히 구씨....또 반함 너무 재밌게 봤어요강추추추추'],
                   [2, '돈없니? 액션씬좀 많이넣어라 명색의 액션영화인데 너무지루하다']], dtype=object)
        '''
        return corpus

    def count_words(self, training_set):
        counts = defaultdict(lambda: [0, 0])
        for doc, point in training_set:
            # 영화리뷰가 text 일때만 카운드
            if self.is_number(doc) is False:
                # 리뷰를 띄어쓰기 단위로 토크나이징
                words = doc.split()
                for word in words:
                    counts[word][0 if point > 3.5 else 1] += 1
        return counts

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def word_probabilities(counts, total_class0, total_class1, k):
        # 단어의 빈도수를 [단어, p(w|긍정), p(w|부정)] 형태로 변환
        return [(w, (class0 + k) / (total_class0 + 2 * k), (class1 + k) / (total_class1 + 2 * k)) for w, (class0, class1) in counts.items()]

    @staticmethod
    def class0_probabilities(word_probs, doc):
        # 별도 토크나이즈 하지 않고 띄어쓰기만
        print(f'코멘트 : {doc}', end='\t\t\t')
        docwords = doc.split()
        # 초기값은 모두 0으로 처리
        log_prob_if_class0 = log_prob_if_class1 = 0.0
        # 모든 단어에 대해 반복
        for word, prob_if_class0, prob_if_class1 in word_probs:
            # 만약 리뷰에 word 가 나타나면 해당 단어가 나올 log 에 확률을 더 해줌
            if word in docwords:
                log_prob_if_class0 += math.log(prob_if_class0)
                log_prob_if_class1 += math.log(prob_if_class1)
            # 만약 리뷰에 word 가 나타나지 않는다면
            # 해당 단어가 나오지 않을 log 에 확률을 더해줌
            # 나오지 않을 확률은 log(1 - 나올 확률) 로 계산
            else:
                log_prob_if_class0 += math.log(1.0 - prob_if_class0)
                log_prob_if_class1 += math.log(1.0 - prob_if_class1)
        prob_if_class0 = math.exp(log_prob_if_class0)
        prob_if_class1 = math.exp(log_prob_if_class1)
        return prob_if_class0 / (prob_if_class0 + prob_if_class1)

    def train(self, training_set):
        print('********** 훈련 시작 **********')
        # 범주0 (긍정) 과 범주1(부정) 문서의 수를 세어줌
        num_class0 = len([1 for point, _ in training_set if point > 7])
        num_class1 = len(training_set) - num_class0
        # train
        word_counts = self.count_words(training_set)
        return self.word_probabilities(word_counts, num_class0, num_class1, self.k)

    def classify(self, word_probs, doc):
        return self.class0_probabilities(word_probs, doc)

    @staticmethod
    def postprocess(point):
        if point > 0.8:
            return f'토마토 싱싱하네요. 평점 : {round(point * 10, 2)}'
        elif point > 0.5:
            return f'토마토 맛없어요. 평점 : {round(point * 10, 2)}'
        else:
            return f'토마토 썩었습니다. 평점 : {round(point * 10, 2)}'


if __name__ == '__main__':
    Solution().hook()
