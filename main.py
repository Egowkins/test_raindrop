from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from Normal import optimization, heigh_search, plotter_maker, np_to_df




def raindrop_collector(for_train, window_size: int = 50000, height_peak=None,
                       rolling_window: int = 1000, df=None,  plot=True):
    """

    :param for_train: датасет для выделения капель
    :param window_size: размер окна для ввода вручную, при отсутствии - назначается эмпирически выверенное
    :param height_peak: высота амплитуды. При отстутствии введенных данных - берется по формуле
    :param rolling_window: начальный размер скользящего окна
    :param plot: true - означает чертить графики в конце работы прогрммы.
    :return: возвращает массив numpy капелек. каждая ячейка содержит df капельки
    """

    if type(window_size) != int:
        raise TypeError('Тип данных должен быть int')

    for_train = optimization(for_train, rolling_window)


    #hardcode для эмпирически выверенного окна (лучшего окна пока выведено не было)

    if height_peak is None:
        height_peak = heigh_search(for_train)


    # находим пики с высотой больше ...
    peaks, i = find_peaks(for_train['Channel A'], height=height_peak, distance=window_size)

    print(str(len(peaks)) + " length of peaks\n")

    #храниние капель
    setup = np.empty((len(for_train),), dtype=object)
    print("Создано хранилище капель")


    #Объявляем глобальную переменную для учета индекса

    global ITERATION

    #поиск пиков и вырез окна
    for i, peak in enumerate(peaks):

        start = max(peak - window_size // 2, 0)
        end = min(peak + window_size // 2, len(for_train))
        # Вырезаем окно для каждого канала
        window = for_train.loc[start:end, ['Time', 'Channel A', 'Channel B', 'Channel C', 'Channel D']].copy()
        # Добавляем окно в массив капелек

        window['ID'] = (i + ITERATION)

        time_values = for_train['Time'].values

        distances = []

        """
        for j in range(0, len(peaks)-1):

            time_diff = time_values[peaks[j+1]] - time_values[peaks[j]]
            distances.append(time_diff)

            print(time_diff)

        distances.append(0)
        
        window['x'] = distances[i]
        1: 
        """

        setup[i] = window

    ITERATION += len(peaks)


    #todo: расстояние между пиками 1,2,3,4



    """
    #пока что чертим 4 капельки
    for i in range(4):
        if plot:
            plotter_maker(setup[i])

    plotter_maker(for_train, peaks)
    """


    df_rainrops = np_to_df(setup, df)
    print(df_rainrops.head(1000))
    print(df_rainrops.tail(1000))


    return df_rainrops


if __name__ == "__main__": #без кавычек)))

    ITERATION = 0


    df = pd.read_csv('venv/20231206-0001.csv', sep=';', decimal=',', low_memory=False)



    WIN_SIZE = 50000

    a = raindrop_collector(df, WIN_SIZE)

    b = raindrop_collector(df, WIN_SIZE, df=a)













