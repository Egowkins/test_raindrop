import os
import re
import scipy.io
import pandas as pd
import numpy as np


def finder():

    path = os.getcwd()

    dir_pattern = re.compile(r"^M\d+$")
    file_pattern = re.compile(r"^M\d+_\d+\.mat$")

    # Список для собранных файлов
    collected_files = []

    # Итерация по элементам в директории
    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        if os.path.isdir(item_path) and dir_pattern.match(item):
            for file in os.listdir(item_path):
                if file_pattern.match(file):
                    collected_files.append(os.path.join(item_path, file))

    # Сортировка списка файлов
    collected_files.sort()
    print(collected_files)

    return collected_files


def aggregate_find(file):
    mat_data = scipy.io.loadmat(file)
    #Определение начальной позиции временного отрезка и его интервала из .mat файла
    Tstart = mat_data['Tstart'][0, 0]
    Tinterval = mat_data['Tinterval'][0, 0]

    # Вычисление количества точек данных
    num_data_points = len(mat_data['A'])

    # Генерация временного ряда
    Time = np.arange(Tstart, Tstart + Tinterval * num_data_points, Tinterval)
    # Создание датафрейма
    df = pd.DataFrame({
        'Time': Time,
        'Channel A': mat_data['A'].ravel(),
        'Channel B': mat_data['B'].ravel(),
        'Channel C': mat_data['C'].ravel(),
        'Channel D': mat_data['D'].ravel()
    })
    df['Time'] = df['Time'].map('{:.10f}'.format)
    df[['Time', 'Channel A', 'Channel B', 'Channel C', 'Channel D']] = df[['Time', 'Channel A', 'Channel B',
                                                                           'Channel C', 'Channel D']].astype(float)
    return df


def excel_creator(excel_file_path: str, final: pd.DataFrame) -> None:
    final.to_excel(excel_file_path, index=False)
