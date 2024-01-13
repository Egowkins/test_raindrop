import pandas as pd
import numpy as np
from Normal import heigh_search, plotter_maker, np_to_df, dt_finder, raindrops_and_peaks
from model import feature_extractor
from cat_model import model_rain
import matplotlib.pyplot as plt
import scipy.io
from file_find import finder
from file_concat import exp_param
import os


def raindrop_collector(for_train, window_size: int = 50000,
                       height_peak=None, plot=True):
    """
    #todo настройка лоу пасс фильтра
    :param for_train: датасет для выделения капель
    :param window_size: размер окна для ввода вручную, при отсутствии - назначается эмпирически выверенное
    :param height_peak: высота амплитуды. При отстутствии введенных данных - берется по квантилю (смотри Normal.heigh_search)
    :param plot: true - означает чертить графики во время работы программы.
    :param low_pass - настройка лоу пасс фильтра, если ничего не было введено - то по умолчанию
    :return: возвращает массив numpy  капелек. каждая ячейка содержит df капельки
    """

    if type(window_size) != int:
        raise TypeError('Тип данных должен быть int')

    if height_peak is None:
        height_peak = heigh_search(for_train)

    setup = raindrops_and_peaks(for_train, height_peak, window_size, butter=True)
    """
    setup2 = raindrops_and_peaks(for_train, height_peak, window_size)
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    
    # Наложите каждый график на соответствующую область
    for i, ax in enumerate(axes.flatten()):
        # Вам нужно указать свои столбцы и параметры для графика в функции plot()

        ax.plot(setup2[i]['Time'], setup2[i]['Channel A'], label='Not_Optimized', color='b')
        plt.grid(True)
        ax.plot(setup[i]['Time'], setup[i]['Channel A'], label='Optimized', color='red')
        plt.grid(True)
        ax.legend()


    # Регулируем расположение графиков
    plt.tight_layout()

    # Отображаем графики
    plt.show()
   """

    df_rainrops = np_to_df(setup)

    setup = None
    setup2 = None

    print(df_rainrops.head(1000))
    print(df_rainrops.tail(1000))

    # вставить функцию поиска расстояний между пиками

    return df_rainrops


if __name__ == "__main__":
    # todo многопоточность
    # todo рефакторинг (сделать все функциями!!!)
    # todo юниттесты
    # todo документация
    final = None
    finder()
    for file in finder():

        mat_data = scipy.io.loadmat(file)
        Tstart = mat_data['Tstart'][0, 0]  # Предполагается, что Tstart это скаляр
        Tinterval = mat_data['Tinterval'][0, 0]  # Предполагается, что Tinterval это скаляр

        # Вычисление количества точек данных
        num_data_points = len(mat_data['A'])  # Или любой другой массив того же размера

        # Генерация временного ряда
        Time = np.arange(Tstart, Tstart + Tinterval * num_data_points, Tinterval)
        # Создание DataFrame
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
        print(df)
        WIN_SIZE = 50000


    #размченный датасет полный капель
        df = raindrop_collector(df, WIN_SIZE)


        print("Начало извлечения расстояний между пиками из выборки")

    #датасет с фичами
    #сюда мерджим результат работы

        features_of_df = dt_finder(df)
        print("Конец извлечения расстояний между пиками из выборки")

        for column in df:

            if column != 'Time' and column != 'ID':
                print(f'Извлечение признаков для колонки {column}')
                features_of_df = feature_extractor(df, features_of_df, column)

        row_to_add = exp_param(file)
        print(row_to_add)
        number_of_rows_to_add = features_of_df.shape[0]

        # *Повторяем строки row_to_add столько раз, сколько строк в features_of_df*
        rows_to_add = pd.concat([row_to_add] * number_of_rows_to_add, ignore_index=True, axis=0)

        # Добавляем rows_to_add слева к features_of_df
        features_of_df = pd.concat([features_of_df, rows_to_add], axis=1)

        if final is None:
            final = features_of_df
        else:
            final = pd.concat([features_of_df, final], ignore_index=True)

    excel_file_path = 'output.xlsx'

    final.to_excel(excel_file_path, index=False)
    final.drop('M', axis=1, inplace=True)
    print(final)
    results = model_rain(final)
    print(results)
