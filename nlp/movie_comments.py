from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import numpy as np
import math
import re
from collections import Counter
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
    def __init__(self):
        self.movie_comments = pd.DataFrame()
        self.file = File(context='./save/')
        self.okt = Okt()

    def hook(self):
        def print_menu():
            print('0. Exit')
            print('1. 전처리 : 텍스트 마이닝 (크롤링)')
            print('2. DF 정형화')  # 1, 2를 합치면 preprocess
            print('3. 토큰화')
            print('4. 임베딩')

            print('5. 긍정 리뷰의 상위 20개 형태소를 시각화하시오.')
            print('6. 부정 리뷰의 상위 20개 형태소를 시각화하시오.')
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
                pass
            elif menu == '4':
                pass
            elif menu == '5':
                pass
            elif menu == '6':
                pass
            elif menu == '7':
                pass

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
        ic(info_movie.sort_values(by=['count'], axis=0, ascending=False))

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

    def tokenization(self):  # (, , ,)
        pass

    def embedding(self):  # [] 벡터 리턴
        pass

    '''
    def review_count(self):
        # ic(df_data.info())

        # 코멘트가 없는 리뷰 데이터(NaN) 제거
        df_reviews = df_data.dropna()
        # 중복 리뷰 제거
        df_reviews = df_reviews.drop_duplicates(['comment'])

        df_reviews.info()
        # ic(df_reviews.head(10))

        # 영화 리스트 확인
        movie_lst = df_reviews.title.unique()
        ic('전체 영화 편수 =', len(movie_lst))
        # ic(movie_lst[:10])

        # 각 영화 리뷰 수 계산
        cnt_movie = df_reviews.title.value_counts()
        # ic(cnt_movie[:20])

        # 각 영화 평점 분석
        info_movie = df_reviews.groupby('title')['score'].describe()
        info_movie.sort_values(by=['count'], axis=0, ascending=False)

        ic(df_reviews.label.value_counts())

        return df_reviews

    def visualize(self):
        df_reviews = self.get_comments()
        top10 = df_reviews.title.value_counts().sort_values(ascending=False)[:10]
        top10_title = top10.index.tolist()
        top10_reviews = df_reviews[df_reviews['title'].isin(top10_title)]

        # ic(top10_title)
        # ic(top10_reviews.info())

        movie_title = top10_reviews.title.unique().tolist()  # -- 영화 제목 추출
        avg_score = {}  # -- {제목 : 평균} 저장
        for t in movie_title:
            avg = top10_reviews[top10_reviews['title'] == t]['score'].mean()
            avg_score[t] = avg

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

        plt.show()

        fig, axs = plt.subplots(5, 2, figsize=(15, 25))
        axs = axs.flatten()

        for title, avg, ax in zip(avg_score.keys(), avg_score.values(), axs):
            num_reviews = len(top10_reviews[top10_reviews['title'] == title])
            x = np.arange(num_reviews)
            y = top10_reviews[top10_reviews['title'] == title]['score']
            ax.set_title('\n%s (%d명)' % (title, num_reviews), fontsize=15)
            ax.set_ylim(0, 10.5, 2)
            ax.plot(x, y, 'o')
            ax.axhline(avg, color='red', linestyle='--')  # -- 평균 점선 나타내기

        plt.show()

        fig, axs = plt.subplots(5, 2, figsize=(15, 25))
        axs = axs.flatten()
        colors = ['pink', 'gold', 'whitesmoke']
        labels = ['1 (8~10점)', '0 (1~4점)', '2 (5~7점)']

        for title, ax in zip(avg_score.keys(), axs):
            num_reviews = len(top10_reviews[top10_reviews['title'] == title])
            values = top10_reviews[top10_reviews['title'] == title]['label'].value_counts()
            ax.set_title('\n%s (%d명)' % (title, num_reviews), fontsize=15)
            ax.pie(values,
                   autopct='%1.1f%%',
                   colors=colors,
                   shadow=True,
                   startangle=90)
            ax.axis('equal')
        plt.show()

    def visualize_morpheme_pos(self):
        df_reviews = self.get_comments()
        pos_reviews = df_reviews[df_reviews['label'] == 1]
        # -- 긍정 리뷰
        pos_reviews['comment'] = pos_reviews['comment'].apply(lambda x: re.sub(r'[^ㄱ-ㅣ가-힝+]', ' ', x))
        pos_comment_nouns = []
        for cmt in pos_reviews['comment']:
            pos_comment_nouns.extend(self.okt.nouns(cmt))  # -- 명사만 추출
        # -- 추출된 명사 중에서 길이가 1보다 큰 단어만 추출
        pos_comment_nouns2 = []
        word = [w for w in pos_comment_nouns if len(w) > 1]
        pos_comment_nouns2.extend(word)
        # ic(pos_comment_nouns2)

        pos_word_count = Counter(pos_comment_nouns2)
        # ic(pos_word_count)

        max = 20
        pos_top_20 = {}
        for word, counts in pos_word_count.most_common(max):
            pos_top_20[word] = counts
            # ic(f'{word} : {counts}')
        plt.figure(figsize=(10, 5))
        plt.title('긍정 리뷰의 단어 상위 (%d개)' % max, fontsize=17)
        plt.ylabel('단어의 빈도수')
        plt.xticks(rotation=70)
        for key, value in pos_top_20.items():
            if key == '영화': continue
            plt.bar(key, value, color='lightgrey')
        plt.show()

    def visualize_morpheme_neg(self):
        df_reviews = self.get_comments()
        # -- 부정 리뷰
        neg_reviews = df_reviews[df_reviews['label'] == 0]
        neg_reviews['comment'] = neg_reviews['comment'].apply(lambda x: re.sub(r'[^ㄱ-ㅣ가-힝+]', ' ', x))
        neg_comment_nouns = []

        for cmt in neg_reviews['comment']:
            neg_comment_nouns.extend(self.okt.nouns(cmt))

        neg_comment_nouns2 = []
        word = [w for w in neg_comment_nouns if len(w) > 1]
        neg_comment_nouns2.extend(word)

        # -- 단어 빈도 계산
        neg_word_count = Counter(neg_comment_nouns2)

        # -- 빈도수가 많은 상위 20개 단어 추출
        max = 20
        neg_top_20 = {}
        for word, counts in neg_word_count.most_common(max):
            neg_top_20[word] = counts
            print(f'{word} : {counts}')

        # -- 그래프 작성
        plt.figure(figsize=(10, 5))
        plt.title('부정 리뷰의 단어 상위 (%d개)' % max, fontsize=17)
        plt.ylabel('단어의 빈도수')
        plt.xticks(rotation=70)
        for key, value in neg_top_20.items():
            if key == '영화': continue
            plt.bar(key, value, color='lightgrey')
        plt.show()
    '''


if __name__ == '__main__':
    Solution().hook()
