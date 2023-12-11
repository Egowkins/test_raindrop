from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from Normal import optimization, heigh_search, plotter_maker


def raindrop_collector(for_train, window_size: int, height_peak=None, window_mean=None, plot=True):
    """

    :param for_train: датасет для выделения капель
    :param window_size: размер окна для ввода вручную, при отсутствии - назначается эмпирически выверенное
    :param height_peak: высота амплитуды. При отстутствии введенных данных - берется по формуле
    :param window_mean: начальный размер скользящего окна
    :param plot: true - означает чертить графики в конце работы прогрммы.
    :return: возвращает массив numpy капелек. каждая ячейка содержит df капельки
    """

    if type(window_size) != int:
        raise TypeError('Тип данных должен быть int')


    #hardcode для эмпирически выверенного окна (лучшего окна пока выведено не было) #TODO алгоритмический подбор окна
    if window_size < 50000:
        window_size = 50000
    if height_peak is None:
        height_peak = heigh_search(for_train)


    # находим пики с высотой больше ...
    peaks, i = find_peaks(for_train['Channel A'], height=height_peak, distance=window_size)

    print(str(len(peaks)) + " length of peaks\n")

    #храниние капель
    setup = np.empty((len(for_train),), dtype=object)
    print("!")

    #поиск пиков и вырез окна
    for i, peak in enumerate(peaks):
        start = max(peak - window_size // 2, 0)
        end = min(peak + window_size // 2, len(for_train))
        # Вырезаем окно для каждого канала
        window = for_train.loc[start:end, ['Time', 'Channel A', 'Channel B', 'Channel C', 'Channel D']].copy()
        # Добавляем окно в массив капелек
        setup[i] = window

    """
    #пока что чертим 4 капельки
    for i in range(4):
        if plot:
            plotter_maker(setup[i])
    """
    plotter_maker(for_train, peaks)

    return setup


if __name__ == "__main__": #без кавычек)))

    df = pd.read_csv('venv/20231206-0001.csv', sep=';', decimal=',', low_memory=False)
    df = optimization(df)

#ручной ввод окна
    WIN_SIZE = 50000
    a = raindrop_collector(df, WIN_SIZE)

    print(a)











