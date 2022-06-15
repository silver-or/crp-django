from konlpy.tag import Okt
from nltk.tokenize import word_tokenize
import nltk
import re
import pandas as pd
from nltk import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from icecream import ic

from context.domains import Reader, File

'''
문장 형태의 문자 데이터를 전처리할 때 많이 사용되는 방법이다. 
말뭉치(코퍼스 corpus)를 어떤 토큰의 단위로 분할하냐에 따라 
단어 집합의 크기, 단어 집합이 표현하는 토크의 형태가 다르게 나타나며 
이는 모델의 성능을 좌지우지하기도 한다. 
이때 텍스트를 토큰의 단위로 분할하는 작업을 토큰화라고 한다. 
토큰의 단위는 보통 의미를 가지는 최소 의미 단위로 선정되며, 
토큰의 단위를 단어로 잡으면 Word Tokenization이라고 하고, 
문장으로 잡으면 Sentence Tokeniazation이라고 한다.

영어는 주로 띄어쓰기 기준으로 나누고, 
한글은 단어 안의 형태소를 최소 의미 단위로 인식해 적용한다.

형태소(形態素, 영어: morpheme)는 언어학에서 의미가 있는 가장 작은 말의 단위이다. → atom
말뭉치(영어: corpus 코퍼스)는 언어학에서 주로 구조를 이루고 있는 텍스트 집합이다.

** atom = 토큰 (동적), atom은 scalar부터 시작한다.
** 용어 정리
    코퍼스 : 말뭉치, 대화, 언어, 원초적인 상태, 비정형 데이터 
            텍스트의 집합. 구조는 집합이다. 집합은 최소 vector부터 시작한다. → 코퍼스가 텍스트보다 더 큰 개념 (소리까지 포함)
    텍스트 : word 또는 sentence (word : 띄어쓰기, sentence : 마침표), 국어사전에 등재 (코퍼스에서 음성 및 무의미한 글자 제거)
    ※ 코퍼스(말)에서 키워드 추출 (불필요한 단어 제거) → 코퍼스를 형태소로 분석
    토큰 : 분할하는 단위.
    워드 : 스페이스바 또는 엔터로 나누어져 있음. 형태소가 결합된 상태 (아침에)
    형태소 : 의미가 있는 워드. 실질 형태소를 의미. (아침 : 실질형태소, 에 : 형식 형태소)

1. Preprocessing : kr-Report_2018.txt 를 읽는다. → 객체화
2. Tokenization : 문자열 (string)을 디치원 벡터 (vector)로 변환
    (char : scalar, string : vector) 1차원인 vector부터 객체이므로 객체는 차원을 가진다.
    0차원 (ㄱ, ㄴ, ㄷ ...) 은 prime (원시 타입)
3. Token Embedding
4. Document Embedding

** 임베딩(embedding) / 리프레젠테이션 (representation)
    자연어(코퍼스)에서 텍스트를 추출한다.
    텍스트를 토큰화해서 단어로 만든다.
    단어를 리스트에 담아 벡터화시키는 일련의 과정을 임베딩이라고 한다.
'''


class Solution(Reader):
    def __init__(self):
        self.okt = Okt()
        self.file = File(context='./data/')

    def hook(self):
        def print_menu():
            print('0. Exit')
            print('1. nltk 다운로드')
            print('2. Preprocessing')
            print('3. Tokenization')
            print('4. Token Embedding')
            print('5. 2018년 삼성사업계획서를 분석해서 워드클라우드를 작성하시오.')
            return input('메뉴 선택 \n')

        while 1:
            menu = print_menu()
            if menu == '0':
                break
            elif menu == '1':
                self.download()
            elif menu == '2':
                self.preprocessing()
            elif menu == '3':
                self.tokenization()
            elif menu == '4':
                self.token_embedding()
            elif menu == '5':
                self.draw_wordcloud()

    def preprocessing(self):
        self.okt.pos("삼성전자 글로벌센터 전자사업부", stem=True)
        file = self.file
        file.fname = 'kr-Report_2018.txt'
        path = self.new_file(file)
        with open(path, 'r', encoding='UTF-8') as f:  # 한글 파일 읽기
            texts = f.read()
        texts = texts.replace('\n', ' ')
        tokenizer = re.compile(r'[^ㄱ-힣]+')  # 한글만 빼고 모두 날림  # 정규식 표현 (r) cf. formatter (f) : 반드시 {} 존재
        # ic(tokenizer.sub(' ', texts))
        return tokenizer.sub(' ', texts)

    def tokenization(self):
        noun_tokens = []
        tokens = word_tokenize(self.preprocessing())  # 토큰을 word 단위로 끊는다.
        # ic(tokens[:100])
        for i in tokens:
            pos = self.okt.pos(i)
            _ = [j[0] for j in pos if j[1] == 'Noun']
            if len(''.join(_)) > 1:
                noun_tokens.append(' '.join(_))
        texts = ' '.join(noun_tokens)
        ic(texts[:100])
        return texts

    def read_stopword(self):
        self.okt.pos("삼성전자 글로벌센터 전자사업부", stem=True)
        file = self.file
        file.fname = 'stopwords.txt'
        path = self.new_file(file)
        with open(path, 'r', encoding='UTF-8') as f:  # 한글 파일 읽기
            texts = f.read()
        return texts

    def token_embedding(self) -> []:
        tokens = self.tokenization()
        stopwords = word_tokenize(self.read_stopword())
        texts = [text for text in tokens if text not in stopwords]
        return texts

    def draw_wordcloud(self):
        _ = self.token_embedding()  # _ : 로컬에서만 사용하는 변수, 임시값, self 걸면 안 됨
        freqtxt = pd.Series(dict(FreqDist(_))).sort_values(ascending=False)  # 해당 단어가 얼마나 자주 나왔는지 판단  # 0, 1, 2 가 아닌 큰 수부터 밑으로 떨어짐
        ic(freqtxt)
        wcloud = WordCloud('./data/D2Coding.ttf', relative_scaling=0.2,
                           background_color='white').generate(" ".join(_))
        plt.figure(figsize=(12, 12))
        plt.imshow(wcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    @staticmethod
    def download():
        nltk.download('punkt')


if __name__ == '__main__':
    Solution().hook()
