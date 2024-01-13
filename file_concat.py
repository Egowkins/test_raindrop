import pandas as pd
import os
import re


def exp_param(filename):

    # Поиск файла с нужным названием
    path = os.getcwd()
    params = pd.read_excel(os.path.join(path, 'exp_parameters.xlsx'))
    params.drop('Notes', axis=1, inplace=True)

    def extract_column_name(file):

        return os.path.basename(file).split('_')[0]

    return params[params['M'] == extract_column_name(filename)]


if __name__ == "__main__":
    #проверка
    exp_param('\PycharmProjects\Stagirovka\M11\M11_01')
