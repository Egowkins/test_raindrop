import pandas as pd
import os
import re

def concatination():

    path = os.getcwd()  # Нужно указать путь к папке, в которой находятся файлы
    combined_dataframe_name = 'combined.csv'
    files = [filename for filename in os.listdir(path) if filename.endswith('.csv') ]
    #if len(files) == 1:
        #return os.path.join(path, files[0])

    # Получаем группы числовых значений из названий файлов
    def extract_numbers(filename):
        parts = re.findall(r'\d+', filename)
        return tuple(map(int, parts))

    # Сортируем файлы по числовому значению ключа
    sorted_files = sorted(files, key=extract_numbers)
    print(sorted_files)

    return sorted_files  # Возвращаем путь к объединенному файлу


