import pandas as pd
from Normal import optimization, heigh_search, plotter_maker, np_to_df, dt_finder, raindrops_and_peaks
from model import feature_extractor
from cat_model import model_rain
import matplotlib.pyplot as plt
from file_concat import concatination, exp_param
import os


def raindrop_collector(for_train, window_size: int = 50000, height_peak=None,
                       rolling_window: int = 750, df=None, plot=True):
    """

    :param for_train: датасет для выделения капель
    :param window_size: размер окна для ввода вручную, при отсутствии - назначается эмпирически выверенное
    :param height_peak: высота амплитуды. При отстутствии введенных данных - берется по квантилю (смотри Normal.heigh_search)
    :param rolling_window: начальный размер скользящего окна (не актуально)
    :param plot: true - означает чертить графики в конце работы прогрммы.
    :return: возвращает массив numpy  капелек. каждая ячейка содержит df капельки
    """

    for_train.dropna(inplace=True)
    for_train = for_train.drop(0)

    if type(window_size) != int:
        raise TypeError('Тип данных должен быть int')


    for_train = optimization(for_train, rolling_window)

    if height_peak is None:
        height_peak = heigh_search(for_train)

    setup = raindrops_and_peaks(for_train, height_peak, window_size, butter=True)
    """
    #setup2 = raindrops_and_peaks(for_train, height_peak, window_size)
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

    # прилепить сюда!!!
    df_rainrops = np_to_df(setup)

    setup = None
    setup2 = None

    print(df_rainrops.head(1000))
    print(df_rainrops.tail(1000))

    # вставить функцию поиска расстояний между пиками

    return df_rainrops


if __name__ == "__main__":
    final = None
    for file in concatination():
        #todo .mat  в .csv мазафака
        df = pd.read_csv(f'{os.getcwd()}' + "\\" + file, sep=';', decimal=',', low_memory=False)
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
        number_of_rows_to_add = features_of_df.shape[0]

        # Повторяем строки row_to_add столько раз, сколько строк в features_of_df
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
