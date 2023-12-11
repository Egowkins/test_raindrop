from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def raindrop_collector(for_train, window_size: int, window_mean: None) -> None:

    if window_size < 25000:
        window_size = 25000


    # находим пики с высотой больше ...
    peaks, i = find_peaks(for_train['Channel A'], height=0.025, distance=window_size)

    #храниние капель
    setup = np.empty((len(for_train),), dtype=object)

    for i, peak in enumerate(peaks):
        start = max(peak - window_size // 2, 0)
        end = min(peak + window_size // 2, len(for_train))

        # Вырезаем окно для каждого канала
        window = for_train.loc[start:end, ['Time', 'Channel A', 'Channel B', 'Channel C', 'Channel D']].copy()

        # Добавляем окно в массив капелек
        setup[i] = window


df = pd.read_csv('venv/20231206-0001.csv', sep=';', decimal=',')

#ручной ввод окна
WIN_SIZE = 10000


raindrop_collector(df, WIN_SIZE)











