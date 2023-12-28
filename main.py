import pandas as pd
from Normal import optimization, heigh_search, plotter_maker, np_to_df, dt_finder, raindrops_and_peaks, semi_optimization
from model import feature_extractor
from cat_model import model_rain
import matplotlib.pyplot as plt


def raindrop_collector(for_train, window_size: int = 50000, height_peak=None,
                       rolling_window: int = 750, df=None, plot=True):
    """

    :param for_train: датасет для выделения капель
    :param window_size: размер окна для ввода вручную, при отсутствии - назначается эмпирически выверенное
    :param height_peak: высота амплитуды. При отстутствии введенных данных - берется по формуле
    :param rolling_window: начальный размер скользящего окна
    :param plot: true - означает чертить графики в конце работы прогрммы.
    :return: возвращает массив numpy капелек. каждая ячейка содержит df капельки
    """

    for_train.dropna(inplace=True)
    for_train = for_train.drop(0)
    #print(for_train)
    if type(window_size) != int:
        raise TypeError('Тип данных должен быть int')

    for_plot = for_train.copy()
    for_plot = semi_optimization(for_plot)

    #for_train = semi_optimization(for_train)
    #for_train = low_pass_filter(for_train)
    for_train = optimization(for_train, rolling_window)
    print(for_train)

    # hardcode для эмпирически выверенного окна (лучшего окна пока выведено не было)
    if height_peak is None:
        height_peak = heigh_search(for_plot)

    # находим пики с высотой больше ...
    setup = raindrops_and_peaks(for_train, height_peak, window_size)
    setup2 = raindrops_and_peaks(for_plot, height_peak, window_size)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Наложите каждый график на соответствующую область
    for i, ax in enumerate(axes.flatten()):
        # Вам нужно указать свои столбцы и параметры для графика в функции plot()
        ax.plot(setup[i]['Time'], setup[i]['Channel A'], label='Optimized', color='red')
        ax.plot(setup2[i]['Time'], setup2[i]['Channel A'], label='Not_Optimized', color='b')
        ax.set_title(f'Plot {i + 1}')
        ax.legend()

    # Регулируем расположение графиков
    plt.tight_layout()

    # Отображаем графики
    plt.show()

    # прилепить сюда!!!
    df_rainrops = np_to_df(setup)
    setup = None

    print(df_rainrops.head(1000))
    print(df_rainrops.tail(1000))

    # вставить функцию поиска расстояний между пиками

    return df_rainrops


if __name__ == "__main__":

    ITERATION = 0
    df = pd.read_csv('venv/20231206-0001.csv', sep=';', decimal=',', low_memory=False)
    WIN_SIZE = 50000


    #размченный датасет полный капель
    df = raindrop_collector(df, WIN_SIZE)


    print("Начало извлечения расстояний между пиками из выборки")
    #датасет с фичами
    #сюда мерджим результат работы tsfresh
    features_of_df = dt_finder(df)
    print("Конец извлечения расстояний между пиками из выборки")

    for column in df:
        #print(column)
        if column != 'Time' and column != 'ID':
            print(f'Извлечение признаков для колонки {column}')
            features_of_df = feature_extractor(df, features_of_df, column)
    print(features_of_df)

    excel_file_path = 'output.xlsx'
    features_of_df.to_excel(excel_file_path, index=False)

    #results = model_rain(features_of_df, columns)
    results = model_rain(features_of_df)

    print(results)












