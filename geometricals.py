import pandas as pd
import numpy as np
from typing import Union
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def semi_max(channel) -> Union[int, float]:
    """
    Функция для поиска величины полумаксимума
    :param channel: Series с данными одной колонки с определенным ID
    :return: Полумаксимум из channel
    """
    return channel.max() / 2


# time = df['time'] where df with ID (means raindrop)
def semi_width(channel: pd.Series, time: pd.Series):
    """
    Поиск ширины полувысоты пика
    :param channel: Series с данными одной колонки с определенным ID
    :param time: Series с данными времени одной колонки с определенным ID
    :return: Ширина полувысоты пика. При определенных условиях - разный подход
    """
    normalized = (channel - channel.min()) / (channel.max() - channel.min())

    peak_indx= np.where(normalized == 1)[0]



    if peak_indx.size == 0:
        return 0
    # Берем первый пик с высотой 1
    peak_index = peak_indx[0]

    #r_plot = normalized.iloc[:peak_index].values <= 0.5
    #l_plot = normalized.iloc[peak_index:].values <= 0.5
    # Ищем точки слева и справа от пика, где амплитуда становится равной 0.5
    indices_left = np.where(normalized.iloc[:peak_index].values <= 0.5)[0]
    indices_right = np.where(normalized.iloc[peak_index:].values <= 0.5)[0]

    if indices_left.any() and indices_right.size == 0:

        #если пик "упирается" справа
        left_index = indices_left[-1]
        print(left_index)
        half_width_time = (time.iloc[peak_index] - (time.iloc[left_index] +
                           time.iloc[left_index])/2) * 2

        return half_width_time, left_index

    elif indices_left.size == 0 and indices_right.any():

        #если пик "упирается" слева
        right_index = peak_index + indices_right[0]
        half_width_time = ((time.iloc[right_index] + time.iloc[right_index - 1])/2 -
                           time.iloc[peak_index]) * 2

        return half_width_time, right_index

    # Если индексы не пусты, продолжаем
    elif indices_left.any() and indices_right.any():
        # Находим первый и последний индексы, соответствующие половинной амплитуде
        left_index = indices_left[-1]
        next_left = left_index + 1
        print(left_index)
        right_index = peak_index + indices_right[0]
        previous_right = right_index - 1

        # Рассчитываем полуширину во времени
        half_width_time = (time.iloc[right_index] + time.iloc[right_index - 1])/2 -\
                          (time.iloc[left_index] + time.iloc[left_index + 1]) / 2

        return half_width_time, left_index

    else:
        print("Не удалось найти индексы для расчета полуширины.")
        return 0


def subpeak(channel, time):
    normalized_series = (channel - channel.min()) / (channel.max() - channel.min())
    peak_1 = 1
    #пик
    #Пики по параметрам
    peaks, _ = find_peaks(normalized_series, height=peak_1*0.3)

    peaks_1, _ = find_peaks(normalized_series, height=peak_1*0.3, distance=800)

    peaks_2, _ = find_peaks(normalized_series, height=peak_1*0.3, prominence=(normalized_series.max()*0.3,
                                                                              normalized_series.max()*0.99))



    #Трешхолдовые пики
    threshold_percent = 11
    threshold = np.percentile(normalized_series, threshold_percent)
    subpeaks = [peak for peak in peaks if normalized_series.iloc[peak] > threshold]

    if len(peaks_2) == 0:
        return 0
    elif len(peaks_2) == 1 and peaks_2[0] == normalized_series[normalized_series == peak_1].index:
        return 0
    else:

        p1_indx = normalized_series[normalized_series == peak_1].index

        time_difference = np.abs(time.values[peaks_2] - time.values[p1_indx[0]])

        time_difference = np.delete(time_difference, np.where(time_difference == 0))
        minimum = time_difference.min()
        return minimum


#done
def mean_median(channel) -> Union[int, float]:
    return channel.mean() - channel.median()


#done
def q10(channel) -> Union[int, float]:

    q10 = np.nanquantile(channel, 0.10)
    return q10


#done
def q25(channel) -> Union[int, float]:

    q25 = np.nanquantile(channel, 0.25)
    return q25


#done
def q75(channel) -> Union[int, float]:

    q75 = np.nanquantile(channel, 0.75)
    return q75



#done
def q90(channel) -> Union[int, float]:

    q90 = np.nanquantile(channel, 0.90)
    return q90


#done
def k_semi_max(channel, time) -> Union[int, float]:
    semi_width_peak = semi_width(channel, time)
    if isinstance(semi_width_peak, tuple):
        semi_width_peak = semi_width_peak[1]
        peak = np.where(channel == channel.max())[0]
        print(f'{semi_width_peak} полуширина пика ')
        if semi_width_peak > peak:
            peak = -peak
            semi_width_peak = -semi_width_peak
        return (channel.max() - semi_max(channel))/(peak - semi_width_peak)
    else:
        return 0





if __name__ == "__main__":
    time_values = pd.Series([1, 2, 3, 4, 5, 6])
    channel_values = pd.Series([0, 1, 2, 3, 4, 5])

    result = semi_width(time_values, channel_values)
    print(result)
