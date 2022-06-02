# https://github.com/datasciencedojo/datasets/blob/master/titanic.csv
from titanic.models import TitanicModel
from titanic.templates import TitanicTemplate
from titanic.views import TitanicView

# MTV 에서 View (Controller)
if __name__ == '__main__':
    view = TitanicView()

    def print_menu():
        print('1. 템플릿 2. 전처리')
        return input('메뉴 선택 : ')

    while 1:
        menu = print_menu()
        if menu == '1':
            print(' #### 1. 템플릿 #### ')
            template = TitanicTemplate(fname='train.csv')
            template.visualize()
        elif menu == '2':
            print(' #### 2. 전처리 #### ')
            model = TitanicModel()
            model.learning(train_fname='train.csv', test_fname='test.csv')
        else:
            break
