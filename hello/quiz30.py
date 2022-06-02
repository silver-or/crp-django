import random
import string

import numpy as np
import pandas as pd
from icecream import ic

from context.models import Model
from hello.domains import memberlist


class Quiz30:

    '''
    데이터프레임 문제 Q02.
    ic| df:     A   B   C
            1   1   2   3
            2   4   5   6
            3   7   8   9
            4  10  11  12
    '''
    @staticmethod
    def quiz30_df_4_by_3() -> object:
        '''
        # 1)
        col = [chr(i) for i in range(65, 68)]
        ls = [[i * 3 + j for j in range(1, 4)] for i in range(4)]
        df = pd.DataFrame(ls, index=range(1, 5), columns=col)
        ic(df)
        return df
        '''
        # 2)
        col = [chr(i) for i in range(65, 68)]
        k = range(1, 5)  # [i for i in range(1, 5)]
        v = [[i * 3 + j for j in range(1, 4)] for i in range(4)]
        d = {i: j for i, j in zip(k, v)}
        df = pd.DataFrame.from_dict(d, orient='index', columns=col)
        ic(df)
        return df

    '''
    데이터프레임 문제 Q03.
    두 자리 정수를 랜덤으로 2행 3열 데이터프레임을 생성
    ic| df:     0   1   2
            0  97  57  52
            1  56  83  80
    '''
    @staticmethod
    def quiz31_rand_2_by_3() -> object:
        '''
        내 풀이)
        ls = [[myRandom(10, 99) for j in range(3)] for i in range(2)]
        df = pd.DataFrame(ls)
        ic(df)
        return df
        '''
        # numpy 이용
        df = pd.DataFrame(np.random.randint(10, 99, size=(2, 3)))
        ic(df)
        return df

    '''
    데이터프레임 문제 Q04.
    국어, 영어, 수학, 사회 4과목을 시험 치른 10명의 학생들의 성적표 작성.
    단 점수 0 ~ 100이고 학생은 랜덤 알파벳 5자리 ID 로 표기

    ic| df:        국어  영어  수학  사회
               lDZid  57  90  55  24
               Rnvtg  12  66  43  11
               ljfJt  80  33  89  10
               ZJaje  31  28  37  34
               OnhcI  15  28  89  19
               claDN  69  41  66  74
               LYawb  65  16  13  20
               QDBCw  44  32   8  29
               PZOTP  94  78  79  96
               GOJKU  62  17  75  49
    '''
    @staticmethod
    def name(chr_size) -> str: return ''.join(random.choice(string.ascii_letters) for i in range(chr_size))

    def quiz32_df_grade(self) -> object:
        subjects = ['국어', '영어', '수학', '사회']
        names = [self.name(chr_size=5) for i in range(10)]
        # scores = [[myRandom(0, 100) for j in range(4)] for i in range(10)]
        scores = np.random.randint(0, 100, size=(10, 4))

        df1 = pd.DataFrame(scores, index=names, columns=subjects)  # original
        ic(df1)

        d = dict(zip(names, scores))
        df2 = pd.DataFrame.from_dict(d, orient='index', columns=subjects)
        ic(df2)

        return df1

    @staticmethod
    def create_df(keys, vals, len):
        return pd.DataFrame([dict(zip(keys, vals)) for _ in range(len)])

    def quiz33_df_loc(self) -> object:
        '''
        subjects = ['국어', '영어', '수학', '사회']
        d = [dict(zip(subjects, np.random.randint(0, 100, size=4))) for _ in range(3)]
        df = pd.DataFrame(d)
        ic(df)

        # 같은 결과 출력
        subjects = ['국어', '영어', '수학', '사회']
        scores = np.random.randint(0, 100, size=(4, 3))
        d = dict(zip(subjects, scores))
        df = pd.DataFrame(d)
        ic(df)
        '''

        df = self.create_df(keys=['국어', '영어', '수학', '사회'],
                            vals=np.random.randint(0, 100, size=4),
                            len=3)
        # ic(df)

        '''
        ic| grade_df:      자바  파이썬  자바스크립트  SQL
                      홍정명   2   92      40   55
                      노홍주  60   81      74   86
                      전종현  80   77      85    4
                      정경준  46   79      64   65
                      양정오  29   34      93   81
                      권혜민  48    7       4   45
                      서성민  26    0      98   89
                      조현국  99   61      47   37
                      김한슬  75   35      82   57
                      김진영  81   56      40   20
                      심민혜  29   99      99   61
                      권솔이  48   77      96    4
                      김지혜  91   53      68   75
                      하진희   7   12      18   66
                      최은아  92   87      90   84
                      최민서   1   17      17   53
                      한성수  84   74      41    5
                      김윤섭  28   64      19   15
                      김승현  45   62      17   80
                      강 민  86   24      22   98
                      최건일  13   77      49   15
                      유재혁  50   56      16   58
                      김아름  51   50      28   12
                      장원종  38   66      30   74
        '''
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
        '''
        subjects = ['자바', '파이썬', '자바스크립트', 'SQL']
        students = memberlist()
        scores = np.random.randint(0, 100, size=(len(students), 4))
        students_scores = {student: score for student, score in zip(students, scores)}
        students_scores_df = pd.DataFrame.from_dict(students_scores, orient='index', columns=subjects)
        # students_scores_df = pd.DataFrame(scores, index=students, columns=subjects)
        model = Model()
        # model.save_model(fname='grade.csv', dframe=students_scores_df)
        '''
        model = Model()
        grade_df = model.new_model('grade.csv')
        ic(grade_df)

        print('Q1. 파이썬의 점수만 출력하시오.')
        python_scores = grade_df.loc[:, '파이썬']
        ic(python_scores)  # type(python_scores): <class 'pandas.core.series.Series'>

        print('Q2. 조현국의 점수만 출력하시오.')
        cho_scores = grade_df.loc['조현국']
        ic(cho_scores)  # type(cho_scores): <class 'pandas.core.series.Series'>

        print('Q3. 조현국의 과목별 점수를 출력하시오.')
        cho_subjects_scores = grade_df.loc[['조현국']]
        ic(cho_subjects_scores)  # type(cho_subjects_scores): <class 'pandas.core.frame.DataFrame'>

        return grade_df

    def quiz34_df_iloc(self) -> None:
        df = self.create_df(keys=['국어', '영어', '수학', '사회'],
                            vals=np.random.randint(0, 100, size=4),
                            len=3)
        # ic(df.iloc[0])  # Series
        '''
        # 첫 번째 학생의 성적표
        ic| df.iloc[0]: 국어    43
                        영어    72
                        수학    71
                        사회    31
                        Name: 0, dtype: int32
        '''
        # ic(df.iloc[[0]])  # DataFrame
        '''
        ic| df.iloc[[0]]:    국어  영어  수학  사회
                            0  63  34  53  67
        '''
        # ic(df.iloc[[0, 1]])
        '''
        ic| df.iloc[[0, 1]]:    국어  영어  수학  사회
                             0  63  34  53  67
                             1  63  34  53  67
        '''

        # ic(df.iloc[:3])
        '''
        ic| df.iloc[:3]:    국어  영어  수학  사회
                         0  54  92  16  18
                         1  54  92  16  18
                         2  54  92  16  18
        '''

        # ic(df.iloc[[True, False, True]])
        '''
        ic| df.iloc[[True, False, True]]:    국어  영어  수학  사회
                                          0  99  33  73  69
                                          2  99  33  73  69
        '''

        # ic(df.iloc[lambda x: x.index % 2 == 0])
        '''
        ic| df.iloc[lambda x: x.index % 2 == 0]:    국어  영어  수학  사회
                                                 0   5  85  23  78
                                                 2   5  85  23  78
        '''

        # ic(df.iloc[0, 1])
        '''
        # 좌표값
        ic| df.iloc[0, 1]: 88
        '''

        # ic(df.iloc[[0, 2], [1, 3]])
        '''
        ic| df.iloc[[0, 2], [1, 3]]:    영어  사회
                                     0  73  21
                                     2  73  21
        '''
        # ic(df.iloc[1:3, 0:3])
        '''
        ic| df.iloc[1:3, 0:3]:    국어  영어  수학
                               1  33  73  44
                               2  33  73  44
        '''

        # ic(df.iloc[:, [True, False, True, False]])
        '''
        ic| df.iloc[:, [True, False, True, False]]:    국어  수학
                                                    0  85  64
                                                    1  85  64
                                                    2  85  64
        '''

        # ic(df.iloc[:, lambda df:[0, 2]])
        '''
        ic| df.iloc[:, lambda df:[0, 2]]:    국어  수학
                                          0  46  36
                                          1  46  36
                                          2  46  36

        '''
        return None

    def quiz35(self) -> str: return None

    def quiz36(self) -> str: return None

    def quiz37(self) -> str: return None

    def quiz38(self) -> str: return None

    def quiz39(self) -> str: return None
