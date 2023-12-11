from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Normal import optimization


def raindrop_collector(for_train, window_size: int, window_mean = None):

    if type(window_size) != int:
        raise TypeError('Тип данных должен быть int')

    #hardcode для эмпирически выверенного окна
    if window_size < 50000:
        window_size = 50000


    # находим пики с высотой больше ...
    peaks, i = find_peaks(for_train['Channel A'], height=0.025, distance=window_size)

    #храниние капель
    setup = np.empty((len(for_train),), dtype=object)
    print("!")

    for i, peak in enumerate(peaks):
        start = max(peak - window_size // 2, 0)
        end = min(peak + window_size // 2, len(for_train))

        # Вырезаем окно для каждого канала
        window = for_train.loc[start:end, ['Time', 'Channel A', 'Channel B', 'Channel C', 'Channel D']].copy()

        # Добавляем окно в массив капелек
        setup[i] = window

    return setup

if __name__ == "__main__": #без кавычек)))

    df = pd.read_csv('venv/20231206-0001.csv', sep=';', decimal=',', low_memory=False)
    df = optimization(df)

#ручной ввод окна
    WIN_SIZE = 10000
    a = raindrop_collector(df, WIN_SIZE)

    print(a)











