from context.domains import Dataset
import pandas as pd


class Model:
    def __init__(self):
        self.ds = Dataset()
        this = self.ds
        this.dname = './data/'  # 읽어올 때
        this.sname = './save/'  # 저장할 때

    def get_sname(self):
        return self.ds.sname

    def new_model(self, fname) -> object:  # 메모리
        this = self.ds
        '''
        index_col=0 해야 기존 index 값이 유지된다.
        0은 column명 중 첫 번째를 의미한다. (배열 구조)
        pd.read_csv('경로/파일명.csv', index_col='index로 지정할 column명') Index 지정
        '''
        return pd.read_csv(f'{this.dname}{fname}', index_col=0)

    def new_dframe(self, fname) -> object:  # 메모리
        this = self.ds
        # pd.read_csv('경로/파일명.csv') Index 지정 안 함
        return pd.read_csv(f'{this.dname}{fname}')

    def save_model(self, fname, dframe):  # 디스크
        this = self.ds
        '''
        풀옵션은 다음과 같다.
        df.to_csv(f'{self.ds.dname}{fname}', sep=',', na_rep='NaN',
                  float_format='%.2f',  # 2 decimal places
                  columns=['ID', 'X2'],  # columns to write
                  index=False)  # do not write index
        '''
        dframe.to_csv(f'{this.sname}{fname}', sep=',', na_rep='NaN')
