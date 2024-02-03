import pandas as pd
from Normal import heigh_search, plotter_maker, np_to_df, dt_finder, raindrops_and_peaks
from model import feature_extractor
from cat_model import model_rain
from file_find import finder, aggregate_find, excel_creator
from file_concat import exp_param


def catboost_decorator(func):
    def wrapper(*args) -> None:
        final = func(*args)
        excel_creator('output.xlsx', final)
        final.drop('M', axis=1, inplace=True)
        print(final)
        results = model_rain(final)

        print(results)
    return wrapper


def raindrop_collector(for_train: pd.DataFrame, window_size: int = 50000,
                       height_peak=None, plot=True) -> pd.DataFrame:
    """
    #todo настройка лоу пасс фильтра
    :param for_train: датасет для выделения капель
    :param window_size: размер окна для ввода вручную, при отсутствии - назначается эмпирически выверенное
    :param height_peak: высота амплитуды. При отстутствии введенных данных - берется по квантилю (смотри Normal.heigh_search)
    :param plot: true -True = создать графики после работы программы.
    :param low_pass - настройка лоу пасс фильтра, если ничего не было введено - то по умолчанию
    :return: возвращает массив numpy  капелек. каждая ячейка содержит df капельки
    """

    if type(window_size) != int:
        raise TypeError('Тип данных должен быть int')

    if height_peak is None:
        height_peak = heigh_search(for_train)

    setup = raindrops_and_peaks(for_train, height_peak, window_size, butter=True)

    if plot:
        setup2 = raindrops_and_peaks(for_train, height_peak, window_size)
        for column in for_train:
            if column != "Time":
                plotter_maker(setup, setup2, column)

    df_rainrops = np_to_df(setup)

    setup = None
    setup2 = None

    return df_rainrops


@catboost_decorator
def file_d(*args) -> pd.DataFrame:
    """
    Функция применяет raindrop_collector и feature_extractor для всех файлов
    :param args: входные аргументы такие же, каки у raindrop_collector
    :return: датасет с признаками, выделенными из всех капель из всех файлов
    """
    final = None

    for file in finder():
        df = aggregate_find(file)
    #размченный датасет полный капель
        df = raindrop_collector(df, *args)
        print("Начало извлечения расстояний между пиками из выборки")
    #датасет с фичами
    #сюда конкатим результат работы
        features_of_df = dt_finder(df)
        print("Конец извлечения расстояний между пиками из выборки")

        for column in df:

            if column != 'Time' and column != 'ID':

                print(f'Извлечение признаков для колонки {column} из файла {file}')
                features_of_df = feature_extractor(df, features_of_df, column)

        row_to_add = exp_param(file)

        number_of_rows_to_add = features_of_df.shape[0]

        # *Повторяем строки row_to_add столько раз, сколько строк в features_of_df*
        rows_to_add = pd.concat([row_to_add] * number_of_rows_to_add, ignore_index=True, axis=0)

        # Добавляем rows_to_add слева к features_of_df
        features_of_df = pd.concat([features_of_df, rows_to_add], axis=1)

        if final is None:
            final = features_of_df
        else:
            final = pd.concat([features_of_df, final], ignore_index=True)

    return final


if __name__ == "__main__":
    # todo юниттесты
    # todo документация

    file_d()







