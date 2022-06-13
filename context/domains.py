# context, fname, train, test, id, label
from abc import ABC
from dataclasses import dataclass
from abc import *
import pandas as pd
import googlemaps
from typing import TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
GooglemapsClient = TypeVar('googlemaps.Client')

# 중복 데이터
# 공용 설정값
@dataclass
class Dataset:
    dname: str
    sname: str
    fname: str
    train: str
    test: str
    id: str
    label: str

    @property
    def dname(self) -> str: return self._dname

    @dname.setter
    def dname(self, dname): self._dname = dname

    @property
    def sname(self) -> str: return self._sname

    @sname.setter
    def sname(self, sname): self._sname = sname

    @property
    def fname(self) -> str: return self._fname

    @fname.setter
    def fname(self, fname): self._fname = fname

    @property
    def train(self) -> str: return self._train

    @train.setter
    def train(self, train): self._train = train

    @property
    def test(self) -> str: return self._test

    @test.setter
    def test(self, test): self._test = test

    @property
    def id(self) -> str: return self._id

    @id.setter
    def id(self, id): self._id = id

    @property
    def label(self) -> str: return self._label

    @label.setter
    def label(self, label): self._label = label


@dataclass
class File(object):
    context: str
    fname: str
    dframe: object

    @property
    def context(self) -> str: return self._context

    @context.setter
    def context(self, context): self._context = context

    @property
    def fname(self) -> str: return self._fname

    @fname.setter
    def fname(self, fname): self._fname = fname

    @property
    def dframe(self) -> str: return self._dframe

    @dframe.setter
    def dframe(self, dframe): self._dframe = dframe


class PrinterBase(metaclass=ABCMeta):
    @abstractmethod
    def dframe(self, this):
        pass


# new_file, csv, xls, json
class ReaderBase(metaclass=ABCMeta):
    @abstractmethod
    def new_file(self, file) -> str:
        pass

    @abstractmethod
    def csv(self, file) -> object:
        pass

    @abstractmethod
    def xls(self, file, header, cols, skiprow) -> object:
        pass

    @abstractmethod
    def json(self, file) -> object:
        pass


# Printer
class Printer(PrinterBase):
    def dframe(self, this):
        pass


# Reader
class Reader(ReaderBase):
    def new_file(self, file) -> str:
        return file.context + file.fname  # file.context는 나중에 클라우드에 올리면 발급됨

    def csv(self, file) -> PandasDataFrame:  # pandas에서 object는 데이터프레임
        csv = pd.read_csv(f'{self.new_file(file)}.csv', encoding='UTF-8', thousands=',')
        print(f'type: {type(csv)}')
        return csv

    def xls(self, file, header, cols, skiprow) -> PandasDataFrame:
        xls = pd.read_excel(f'{self.new_file(file)}.xls', header=header, usecols=cols, skiprows=[skiprow])
        print(f'type: {type(xls)}')
        return xls

    def json(self, file) -> PandasDataFrame:
        return pd.read_json(f'{self.new_file(file)}.json', encoding='UTF-8')

    @staticmethod
    def gmaps() -> GooglemapsClient:
        return googlemaps.Client(key='')

    @staticmethod
    def print(this):
        print('*' * 100)
        print(f'1. Target type\n {type(this)}')
        print(f'2. Target column\n {this.columns}')
        print(f'3. Target top 1개 행\n {this.head(1)}')
        print(f'4. Target bottom 1개 행\n {this.tail(1)}')
        print(f'5. Target null 의 갯수\n {this.isnull().sum()}개')
        print('*' * 100)
